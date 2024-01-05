"""
Implementing classification Histories class for storing training and validation metrics during training

There will likely be overlap with the RBP implementation at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/metrics.py

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
import os
import random
from typing import Dict, List, Tuple, Optional, Any, NamedTuple

import pandas
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, \
    r2_score
import torch

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
                self._prediction_history[subject].append(float(prediction.cpu()))
            else:
                self._prediction_history[subject] = [float(prediction.cpu())]

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
    def history(self):
        # todo: consider returning values as tuples
        return self._history

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
        df.round({col: decimals for col in epochs_column_names})

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

    # -----------------
    # Metrics
    # todo: make tests and add more metrics
    # todo: add Concordance Correlation Coefficient
    # -----------------
    # Regression metrics
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

    # Classification metrics
    @staticmethod
    @classification_metric
    def auc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return roc_auc_score(y_true=y_true.cpu(), y_score=y_pred.cpu())


if __name__ == "__main__":
    # Define splits
    my_splits = (
        ({"sex": ("female",), "cognition": ("hc", "mci")},  # Inclusion criteria  todo: NamedTuple
         {"education": (Criterion((1, 2, 3)), Criterion((4, 5, 6))),
          "age": (Criterion(("young",)), (Criterion("old",)), Criterion(("young", "old"))),
          "cognition": (Criterion(("hc",)), Criterion(("mci",)))}),
        ({"sex": ("male",), "cognition": ("mci", "ad")},  # Inclusion criteria
         {"education": (Criterion((1, 2)), Criterion((3, 4)), Criterion((5, 6))),
          "age": (Criterion(("young",)), Criterion(("old",))),
          "cognition": (Criterion(("ad",)), Criterion(("mci",)))})
    )

    # Define subjects
    my_subjects = []
    my_sexes = ("male", "female")
    my_cognitions = ("mci", "hc", "ad")
    my_ages = ("old", "young")
    education = (1, 2, 3, 4, 5, 6)

    for i_ in range(10):
        for j_ in range(13):
            my_details = {"sex": random.choice(my_sexes), "cognition": random.choice(my_cognitions),
                          "age": random.choice(my_ages), "education": random.choice(education)}
            my_subjects.append(Subject(f"P{j_}", f"D{i_}", details=my_details))
    my_subjects = tuple(my_subjects)  # type: ignore[assignment]

    # Create object for tracking metrics
    my_history = Histories(metrics="regression", splits=my_splits)

    # Pretend to do predictions
    batch_size = len(my_subjects)
    my_predictions = torch.rand(size=(batch_size, 1), dtype=torch.float)
    my_targets = torch.randint(low=0, high=2, size=(batch_size, 1), dtype=torch.float)

    # Pretend that the batch is done
    my_history.store_batch_evaluation(y_pred=my_predictions, y_true=my_targets, subjects=my_subjects)

    # Pretend that the epoch is done
    my_history.on_epoch_end(verbose=True, verbose_sub_groups=True)
