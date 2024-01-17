"""
Implementing classification Histories class for storing training and validation metrics during training

There will likely be overlap with the RBP implementation at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/metrics.py

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, NamedTuple

import pandas
from matplotlib import pyplot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, \
    r2_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score, log_loss
import torch
from torch import nn

from cdl_eeg.data.data_split import Subject
from cdl_eeg.data.subject_split import Criterion, filter_subjects, make_subject_splits

# Type hints
_SPLIT = Tuple[Dict[str, tuple], Dict[str, Tuple[Criterion, ...]]]  # type: ignore[type-arg]


# ----------------
# Convenient decorators
# ----------------
def regression_metric(func):
    setattr(func, "_is_regression_metric", True)
    return func


def classification_metric(func):
    setattr(func, "_is_classification_metric", True)
    return func


def multiclass_classification_metric(func):
    setattr(func, "_is_multiclass_classification_metric", True)
    return func


# ----------------
# Convenient small classes
# ----------------
class YYhat(NamedTuple):
    """Tuple for storing target and prediction"""
    y_true: torch.Tensor
    y_pred: torch.Tensor


# ----------------
# Classes
# ----------------
class Histories:
    """
    Class for keeping track of all metrics during training. Works for both classification and regression

    TODO: It is strange to store outputs and targets, and compute metrics at the end of the epoch for correlation
        metrics

    Examples
    --------
    >>> Histories.get_available_classification_metrics()
    ('auc',)
    >>> Histories.get_available_regression_metrics()
    ('mae', 'mape', 'mse', 'pearson_r', 'spearman_rho')
    >>> Histories.get_available_multiclass_classification_metrics()
    ('acc', 'balanced_acc', 'ce_loss', 'kappa', 'mcc')
    """

    __slots__ = ("_history", "_prediction_history", "_splits_histories", "_epoch_y_pred", "_epoch_y_true",
                 "_epoch_subjects", "_name")

    def __init__(self, metrics, name=None, splits: Optional[Tuple[_SPLIT, ...]] = None):
        """
        Initialise

        Parameters
        ----------
        metrics : str | tuple[str, ...]
            The metrics to use. If 'str', it must either be 'regression' or 'classification', specifying that all
            available regression/classification metrics should be used
        name : str, optional
            May be used for the printing of the metrics
        splits : Tuple[_SPLIT, ...], optional
            Splits for computing metrics per subgroup (see the PDF file for visual example of a single _SPLIT). Each
            split has inclusion criteria, coupled with split selections and their split criteria (condition in the
            figure).
        """
        # Maybe set metrics
        if metrics == "regression":
            metrics = self.get_available_regression_metrics()
        elif metrics == "classification":
            metrics = self.get_available_classification_metrics()
        elif metrics == "multiclass_classification":
            metrics = self.get_available_multiclass_classification_metrics()

        # ----------------
        # Input checks
        # ----------------
        # Todo

        # ----------------
        # Set attributes
        # ----------------
        self._name = name

        # ----------------
        # Create history dictionaries
        # ----------------
        # The "normal" one
        self._history: Dict[str, List[float]] = {f"{metric}": [] for metric in metrics}

        # For storing all predictions .
        self._prediction_history: Dict[Subject, List[float]] = dict()

        # Histories per subgroup
        splits_histories: Optional[List[Tuple[Dict[str, Tuple[Any, ...]],
                                              Dict[str, Dict[Criterion, Dict[str, List[float]]]]]]]
        if splits is not None:
            splits_histories = []
            # Loop through all desired splits
            for split in splits:
                # Extract inclusion criteria for the current split
                split_inclusion_criteria = split[0]  # todo: consider named tuple

                # Extract what to split on
                split_selections = split[1]

                # Initialise history dictionary for all conditions and metrics
                split_history: Dict[str, Dict[Criterion, Dict[str, List[float]]]] = dict()
                for split_selection, criteria in split_selections.items():
                    split_history[split_selection] = {criterion: {f"{metric}": [] for metric in metrics}
                                                      for criterion in criteria}

                # Append initialised history to main list
                splits_histories.append((split_inclusion_criteria, split_history))  # todo: again, consider NamedTuple
        else:
            splits_histories = None

        self._splits_histories = splits_histories  # todo: consider renaming to e.g. something with subgroup histories

        # ----------------
        # Initialise epochs predictions and targets.
        # They will be updated for each batch
        # ----------------
        self._epoch_y_pred: List[torch.Tensor] = []
        self._epoch_y_true: List[torch.Tensor] = []
        self._epoch_subjects: List[Subject] = []  # todo: do we really need this?

    def store_batch_evaluation(self, y_pred, y_true, subjects=None):
        """
        Store the prediction, targets, and maybe the corresponding subjects. Should be called for each batch

        Parameters
        ----------
        y_pred : torch.Tensor
        y_true : torch.Tensor
        subjects : tuple[Subject, ...], optional

        Returns
        -------
        None
        """
        self._epoch_y_pred.append(y_pred)
        self._epoch_y_true.append(y_true)

        # Store prediction in predictions history
        for prediction, subject in zip(y_pred, subjects):
            if subject in self._prediction_history:
                if prediction.size()[0] > 1:
                    _prediction = tuple(float(pred) for pred in prediction.cpu().tolist())
                elif prediction.size()[0] == 1:
                    _prediction = float(prediction)  # type: ignore[assignment]
                else:
                    raise ValueError("This should never happen")
                self._prediction_history[subject].append(_prediction)
            else:
                if prediction.size()[0] > 1:
                    _prediction = tuple(float(pred) for pred in prediction.cpu().tolist())
                elif prediction.size()[0] == 1:
                    _prediction = float(prediction)
                else:
                    raise ValueError("This should never happen")
                self._prediction_history[subject] = [_prediction]

        # Store the corresponding subjects, if provided
        if subjects is not None:
            self._epoch_subjects.extend(subjects)

    def on_epoch_end(self, verbose=True, verbose_sub_groups=False) -> None:
        """Updates the metrics, and should be called after each epoch"""
        self._update_metrics()
        if verbose:
            self._print_newest_metrics()
        if verbose_sub_groups and self._splits_histories is not None:
            self._print_newest_subgroups_metrics()

    def _print_newest_metrics(self) -> None:
        """Method for printing the newest metrics"""
        # todo: printing?

        for i, (metric_name, metric_values) in enumerate(self.history.items()):
            if i == len(self.history) - 1:
                if self._name is None:
                    print(f"{metric_name}: {metric_values[-1]:.3f}")
                else:
                    print(f"{self._name}_{metric_name}: {metric_values[-1]:.3f}")
            else:
                if self._name is None:
                    print(f"{metric_name}: {metric_values[-1]:.3f}\t\t", end="")
                else:
                    print(f"{self._name}_{metric_name}: {metric_values[-1]:.3f}\t\t", end="")

    def _print_newest_subgroups_metrics(self):
        if self._splits_histories is not None:
            for i, split in enumerate(self._splits_histories):
                print(f"\n----- Details for split {i} -----")
                print("Inclusion criteria:")
                # todo: mypy complaining?
                # Print who to include
                for selection, condition in split[0].items():  # type: ignore
                    # E.g. selection = "dataset", condition = ("cau_eeg",)
                    print(f"\t{selection.capitalize()} must be in: {condition}")
                    # todo: this is where you left...

                # Loop through to get all metrics and print the newest ones
                for split_selection, criteria_performance in split[1].items():
                    print(f"\n\tDomain: {split_selection}")
                    for criterion, performance in criteria_performance.items():
                        print(f"\t\tSub-group: {criterion}")
                        for j, (metric_name, metric_values) in enumerate(performance.items()):
                            # todo: this looks bad...
                            if j == len(performance) - 1:
                                if self._name is None:
                                    print(f"{metric_name}: {metric_values[-1]:.3f}")
                                else:
                                    print(f"{self._name}_{metric_name}: {metric_values[-1]:.3f}")
                            elif j == 0:
                                if self._name is None:
                                    print(f"\t\t\t{metric_name}: {metric_values[-1]:.3f}\t\t", end="")
                                else:
                                    print(f"\t\t\t{self._name}_{metric_name}: {metric_values[-1]:.3f}\t\t", end="")
                            else:
                                if self._name is None:
                                    print(f"{metric_name}: {metric_values[-1]:.3f}\t\t", end="")
                                else:
                                    print(f"{self._name}_{metric_name}: {metric_values[-1]:.3f}\t\t", end="")

    def _update_metrics(self):
        # Concatenate torch tenors
        y_pred = torch.cat(self._epoch_y_pred, dim=0)
        y_true = torch.cat(self._epoch_y_true, dim=0)

        # -------------
        # Update all metrics of the 'normal' history dict
        # -------------
        for metric, hist in self._history.items():
            hist.append(self._compute_metric(metric=metric, y_pred=y_pred, y_true=y_true))

        # -------------
        # (Maybe) update all metrics of all subgroups
        # -------------
        if self._splits_histories is not None:
            # Make dictionary containing subjects combined with the prediction and target
            subjects_pred_and_true = {subject: YYhat(y_true=y, y_pred=y_hat) for subject, y_hat, y
                                      in zip(self._epoch_subjects, y_pred, y_true)}

            # Loop through all splits
            for split in self._splits_histories:
                # Filter subjects
                inclusion_criteria = split[0]
                filtered_subjects = filter_subjects(subjects=tuple(self._epoch_subjects),
                                                    inclusion_criteria=inclusion_criteria)

                # Split subjects
                split_selections = {selection: tuple(criteria_performance.keys())
                                    for selection, criteria_performance in split[1].items()}
                subjects_splits = make_subject_splits(subjects=filtered_subjects, splits=split_selections)

                # Loop through all subgroups
                for selection, criteria_performance in split[1].items():
                    for criterion, performance in criteria_performance.items():
                        for metric, metric_value in performance.items():
                            # Extract the subgroup
                            sub_group_subjects = subjects_splits[selection][criterion]

                            # Extract their predictions and targets
                            sub_group_y_pred = torch.cat([subjects_pred_and_true[subject].y_pred
                                                          for subject in sub_group_subjects], dim=0)
                            sub_group_y_true = torch.cat([subjects_pred_and_true[subject].y_true
                                                          for subject in sub_group_subjects], dim=0)

                            # Compute metrics for the subgroup and store it
                            metric_value.append(self._compute_metric(metric=metric, y_pred=sub_group_y_pred,
                                                                     y_true=sub_group_y_true))

        # -------------
        # Remove the epoch histories
        # -------------
        self._epoch_y_pred = []
        self._epoch_y_true = []
        self._epoch_subjects = []

    @classmethod
    def _compute_metric(cls, metric: str, *, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Method for computing the specified metric"""
        return getattr(cls, metric)(y_pred=y_pred, y_true=y_true)

    # -----------------
    # Properties
    # -----------------
    @property
    def name(self) -> str:
        return "UNNAMED" if self._name is None else self._name

    @property
    def history(self):
        # todo: consider returning values as tuples
        return self._history

    @property
    def newest_metrics(self):
        return {metric_name: performance[-1] for metric_name, performance in self._history.items()}

    # -----------------
    # Methods for saving
    # -----------------
    def save_prediction_history(self, history_name, path, decimals=3):
        """
        Method for saving the predictions in a csv file

        Parameters
        ----------
        history_name : str
        path : str
        decimals : int
            Number of decimal places for storing the predictions

        Returns
        -------
        None
        """
        # Sanity check
        num_epochs = len(tuple(self._prediction_history.values())[0])  # type: ignore
        assert all(len(predictions) == num_epochs for predictions in self._prediction_history.values())

        # Create pandas dataframe with the prediction histories
        epochs_column_names = [f"epoch{i+1}" for i in range(num_epochs)]
        df = pandas.DataFrame.from_dict(self._prediction_history, orient="index", columns=epochs_column_names)

        # Add dataset and subject ID
        df.insert(loc=0, value=tuple(subject.subject_id for subject in self._prediction_history),  # type: ignore
                  column="sub_id")
        df.insert(loc=0, value=tuple(subject.dataset_name for subject in self._prediction_history),  # type: ignore
                  column="dataset")

        # Drop the index
        df.reset_index(inplace=True, drop=True)

        # Round the predictions
        df = df.round({col: decimals for col in epochs_column_names})

        # Save csv file
        df.to_csv(os.path.join(path, f"{history_name}.csv"), index=False)

    # -----------------
    # Methods for getting the available metrics
    # -----------------
    @classmethod
    def get_available_regression_metrics(cls):
        """Get all regression metrics available for the class. The regression metric must be a method
        decorated by @regression_metric to be properly registered"""
        # Get all regression metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a regression metric
            if callable(attribute) and getattr(attribute, "_is_regression_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    @classmethod
    def get_available_classification_metrics(cls):
        """Get all classification metrics available for the class. The classification metric must be a method
        decorated by @classification_metric to be properly registered"""
        # Get all classification metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a classification metric
            if callable(attribute) and getattr(attribute, "_is_classification_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    @classmethod
    def get_available_multiclass_classification_metrics(cls):
        """Get all multiclass classification metrics available for the class. The classification metric must be a method
        decorated by @multiclass_classification_metric to be properly registered"""
        # Get all classification metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a classification metric
            if callable(attribute) and getattr(attribute, "_is_multiclass_classification_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    # -----------------
    # Regression metrics
    # todo: make tests and add more metrics
    # todo: add Concordance Correlation Coefficient
    # -----------------
    @staticmethod
    @regression_metric
    def mse(y_pred: torch.Tensor, y_true: torch.Tensor):
        return mean_squared_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric
    def mae(y_pred: torch.Tensor, y_true: torch.Tensor):
        return mean_absolute_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric
    def mape(y_pred: torch.Tensor, y_true: torch.Tensor):
        return mean_absolute_percentage_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric
    def pearson_r(y_pred: torch.Tensor, y_true: torch.Tensor):
        # Removing redundant dimension may be necessary
        if y_true.dim() == 2:
            y_true = torch.squeeze(y_true, dim=1)
        if y_pred.dim() == 2:
            y_pred = torch.squeeze(y_pred, dim=1)

        # Compute and return
        return pearsonr(x=y_true.cpu(), y=y_pred.cpu())[0]

    @staticmethod
    @regression_metric
    def spearman_rho(y_pred: torch.Tensor, y_true: torch.Tensor):
        # Removing redundant dimension may be necessary
        if y_true.dim() == 2:
            y_true = torch.squeeze(y_true, dim=1)
        if y_pred.dim() == 2:
            y_pred = torch.squeeze(y_pred, dim=1)

        # Compute and return
        return spearmanr(a=y_true.cpu(), b=y_pred.cpu())[0]

    @staticmethod
    @regression_metric
    def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor):
        return r2_score(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    # -----------------
    # Classification metrics
    # -----------------
    @staticmethod
    @classification_metric
    def auc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return roc_auc_score(y_true=y_true.cpu(), y_score=y_pred.cpu())

    # -----------------
    # Multiclass classification metrics
    #
    # Note that we assume the logits, not the actual probabilities from softmax
    # -----------------
    @staticmethod
    @multiclass_classification_metric
    def acc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return accuracy_score(y_pred=y_pred.cpu().argmax(dim=-1), y_true=y_true.cpu())

    @staticmethod
    @multiclass_classification_metric
    def balanced_acc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return balanced_accuracy_score(y_pred=y_pred.cpu().argmax(dim=-1), y_true=y_true.cpu())

    @staticmethod
    @multiclass_classification_metric
    def mcc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return matthews_corrcoef(y_pred=y_pred.cpu().argmax(dim=-1), y_true=y_true.cpu())

    @staticmethod
    @multiclass_classification_metric
    def kappa(y_pred: torch.Tensor, y_true: torch.Tensor):
        return cohen_kappa_score(y1=y_pred.cpu().argmax(dim=-1), y2=y_true.cpu())

    @staticmethod
    @classification_metric
    def auc_ovo(y_pred: torch.Tensor, y_true: torch.Tensor):
        return roc_auc_score(y_true=y_true.cpu(), y_score=torch.softmax(y_pred, dim=-1).cpu(), multi_class="ovo")

    @staticmethod
    @classification_metric
    def auc_ovr(y_pred: torch.Tensor, y_true: torch.Tensor):
        return roc_auc_score(y_true=y_true.cpu(), y_score=torch.softmax(y_pred, dim=-1).cpu(), multi_class="ovr")

    @staticmethod
    @multiclass_classification_metric
    def ce_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
        with torch.no_grad():
            performance = nn.CrossEntropyLoss(reduction='mean')(y_pred, y_true).cpu()
        return float(performance)


# ----------------
# Functions
# ----------------
def save_discriminator_histories_plots(path, histories):
    """
    Function for saving domain discriminator histories plots

    Parameters
    ----------
    path : str
    histories : Histories | tuple[Histories, ...]

    Returns
    -------
    None
    """
    # Maybe just convert to tuple  todo: check that it is Histories obejct
    if not isinstance(histories, tuple):
        histories = (histories,)

    # ----------------
    # Loop through all metrics
    # ----------------
    # Get all available metrics
    all_metrics = []
    for history in histories:
        all_metrics.extend(list(history.history.keys()))
    all_metrics = set(all_metrics)  # Keep unique ones only

    for metric in all_metrics:
        pyplot.figure(figsize=(12, 6))

        for history in histories:
            pyplot.plot(range(1, len(history.history[metric]) + 1), history.history[metric], label=history.name)

        # ------------
        # Plot cosmetics
        # ------------
        font_size = 15

        pyplot.title(f"Performance ({metric.capitalize()})", fontsize=font_size + 5)
        pyplot.xlabel("Epoch", fontsize=font_size)
        pyplot.ylabel(metric.capitalize(), fontsize=font_size)
        pyplot.tick_params(labelsize=font_size)
        pyplot.legend(fontsize=font_size)
        pyplot.grid()

        # Save figure and close it
        pyplot.savefig(os.path.join(path, f"discriminator_{metric}.png"))

        pyplot.close()


def save_histories_plots(path, *, train_history=None, val_history=None, test_history=None, test_estimate=None):
    """
    Function for saving histories plots

    Parameters
    ----------
    path : str
    train_history : Histories
    val_history : Histories
    test_history : Histories
    test_estimate : Histories

    Returns
    -------
    None
    """
    # If no history object is passed, a warning is raised and None is returned (better to do nothing than potentially
    # ruin an experiment with an unnecessary error)
    if all(history is None for history in (train_history, val_history, test_history, test_estimate)):
        warnings.warn("No history object was passed, skip saving histories plots...")
        return None

    # ----------------
    # Loop through all metrics
    # ----------------
    # Get all available metrics
    all_metrics = set(tuple(train_history.history.keys()) + tuple(val_history.history.keys())
                      + tuple(test_estimate.history.keys()) + tuple(test_history.history.keys()))

    for metric in all_metrics:
        pyplot.figure(figsize=(12, 6))

        # Maybe plot training history
        if train_history is not None:
            pyplot.plot(range(1, len(train_history.history[metric]) + 1), train_history.history[metric],
                        label="Train", color="blue")

        # Maybe plot validation history
        if val_history is not None:
            pyplot.plot(range(1, len(val_history.history[metric]) + 1), val_history.history[metric], label="Validation",
                        color="orange")

        # Maybe plot validation history
        if test_history is not None:
            pyplot.plot(range(1, len(test_history.history[metric]) + 1), test_history.history[metric],
                        label="Test", color="green")

        # Maybe plot test history
        if test_estimate is not None:
            # The test estimate metric will just be a line across the figure. Need to get stop x value
            # Start value
            x_max = []
            if train_history is not None:
                x_max.append(len(train_history.history[metric]))
            if val_history is not None:
                x_max.append(len(val_history.history[metric]))
            if test_history is not None:
                x_max.append(len(test_history.history[metric]))
            x_stop = max(x_max) if x_max else 2

            # Plot
            pyplot.plot((1, x_stop), (test_estimate.history[metric], test_estimate.history[metric]),
                        label="Test estimate", color="green")

        # ------------
        # Plot cosmetics
        # ------------
        font_size = 15

        pyplot.title(f"Performance ({metric.capitalize()})", fontsize=font_size+5)
        pyplot.xlabel("Epoch", fontsize=font_size)
        pyplot.ylabel(metric.capitalize(), fontsize=font_size)
        pyplot.tick_params(labelsize=font_size)
        pyplot.legend(fontsize=font_size)
        pyplot.grid()

        # Save figure and close it
        pyplot.savefig(os.path.join(path, f"{metric}.png"))

        pyplot.close()


def is_improved_model(old_metrics, new_metrics, main_metric):
    """
    Function for checking if the new set of metrics is evaluated as better than the old metrics, defined by a main
    metric

    Parameters
    ----------
    old_metrics : dict[str, float] | None
    new_metrics : dict[str, float]
    main_metric : str

    Returns
    -------
    bool

    Examples
    --------
    >>> my_old_metrics = {"mae": 3, "mse": 7.7, "mape": 0.3, "pearson_r": 0.9, "spearman_rho": 0.8, "r2_score": -3.1}
    >>> my_new_metrics = {"mae": 3.2, "mse": 4.4, "mape": 0.2, "pearson_r": 0.7, "spearman_rho": 0.9, "r2_score": -3.05}
    >>> is_improved_model(None, my_new_metrics, main_metric="mae")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="mae")
    False
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="mse")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="mape")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="pearson_r")
    False
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="spearman_rho")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="r2_score")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="not_a_metric")  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: Expected the metric to be in ('pearson_r', 'spearman_rho', 'r2_score', 'auc', 'mae', 'mse', 'mape'),
    but found 'not_a_metric'
    """
    # If the old metrics is None, it means that this is the first epoch
    if old_metrics is None:
        return True

    # Define the metrics where the higher, the better, and the lower, the better
    higher_is_better = ("pearson_r", "spearman_rho", "r2_score", "auc")
    lower_is_better = ("mae", "mse", "mape")

    # ----------------
    # Evaluate
    # ----------------
    if main_metric in higher_is_better:
        return old_metrics[main_metric] < new_metrics[main_metric]
    elif main_metric in lower_is_better:
        return old_metrics[main_metric] > new_metrics[main_metric]
    else:
        raise ValueError(f"Expected the metric to be in {higher_is_better + lower_is_better}, but found "
                         f"'{main_metric}'")


if __name__ == "__main__":

    scores_ = torch.tensor([
        [0.1, 0.2, 0.7],
        [0.5, 0.2, 0.3],
        [0.3, 0.4, 0.3],
        [0.5, 0.4, 0.1]
    ], dtype=torch.float)

    targets_ = torch.tensor([
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=torch.float).argmax(dim=-1)
    print(torch.sum(scores_, dim=-1))

    print(f"Multiclass performance (sklearn): {log_loss(['b', 'a', 'b', 'b'], scores_)}")
    print(f"Multiclass performance (pytorch): {nn.CrossEntropyLoss(reduction='mean')(scores_, targets_)}")
