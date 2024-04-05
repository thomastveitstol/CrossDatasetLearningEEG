"""
Implementing classification Histories class for storing training and validation metrics during training

There will likely be overlap with the RBP implementation at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/metrics.py

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
import itertools
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Union

import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, \
    r2_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
import torch
from torch import nn

from cdl_eeg.data.subject_split import Subject


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


def distance_metric(func):
    """Methods labelled with this decorator should compute the distance between two distributions"""
    setattr(func, "_is_distance_metric", True)
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

    Keep in mind that it is strange to store outputs and targets, and compute metrics at the end of the epoch for
    correlation metrics in particular, as the outputs are computed with different weights.

    Examples
    --------
    >>> Histories.get_available_classification_metrics()
    ('auc',)
    >>> Histories.get_available_regression_metrics()
    ('mae', 'mape', 'mse', 'pearson_r', 'r2_score', 'spearman_rho')
    >>> Histories.get_available_multiclass_classification_metrics()
    ('acc', 'auc_ovo', 'auc_ovr', 'balanced_acc', 'ce_loss', 'kappa', 'mcc')
    >>> Histories.get_available_metrics()  # doctest: +NORMALIZE_WHITESPACE
    ('auc', 'acc', 'auc_ovo', 'auc_ovr', 'balanced_acc', 'ce_loss', 'kappa', 'mcc', 'mae', 'mape', 'mse', 'pearson_r',
     'r2_score', 'spearman_rho')
    """

    __slots__ = ("_history", "_prediction_history", "_subgroup_histories", "_epoch_y_pred", "_epoch_y_true",
                 "_epoch_subjects", "_name")

    def __init__(self, metrics, *, name=None, splits):
        """
        Initialise

        Parameters
        ----------
        metrics : str | tuple[str, ...]
            The metrics to use. If 'str', it must either be 'regression' or 'classification', specifying that all
            available regression/classification metrics should be used
        name : str, optional
            May be used for the printing of the metrics
        splits : dict[str, typing.Any] | None
            Splits for computing metrics per subgroup. Each 'split' must be an attribute of the Subject objects passed
            to 'store_batch_evaluation'. Note that this should also work for any class inheriting from Subject, allowing
            for more customised subgroup splitting
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
        # Check if all metrics are implemented
        if not all(metric in self.get_available_metrics() for metric in metrics):
            raise ValueError(f"The following metrics were not recognised: "
                             f"{set(metric for metric in metrics if metric not in self.get_available_metrics() )}. The "
                             f"allowed ones are: {self.get_available_metrics()}")

        # Type check the name
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Unrecognised name type: {type(name)}")

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
        self._prediction_history: Dict[Subject, List[List[Union[float, Tuple[float, ...]]]]] = dict()

        # Histories per subgroup. Not doing any check for legal split names, as the subjects may inherit from Subject
        # and thus have additional attributes
        # Could e.g. be {dataset_name: {D1: {metric1: [val1, val2, val3]}}}
        _sub_hist: Optional[Dict[str, Dict[Any, Dict[str, List[Union[float]]]]]]
        if splits is not None:
            _sub_hist = {split: {sub_group: {metric: [] for metric in metrics} for sub_group in sub_groups}
                         for split, sub_groups in splits.items()}
        else:
            _sub_hist = None
        self._subgroup_histories = _sub_hist

        # ----------------
        # Initialise epochs predictions and targets.
        # They will be updated for each batch
        # ----------------
        self._epoch_y_pred: List[torch.Tensor] = []
        self._epoch_y_true: List[torch.Tensor] = []
        self._epoch_subjects: List[Subject] = []

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
                self._prediction_history[subject][-1].append(_prediction)
            else:
                if prediction.size()[0] > 1:
                    _prediction = tuple(float(pred) for pred in prediction.cpu().tolist())
                elif prediction.size()[0] == 1:
                    _prediction = float(prediction)  # type: ignore[assignment]
                else:
                    raise ValueError("This should never happen")
                self._prediction_history[subject] = [[_prediction]]

        # Store the corresponding subjects, if provided
        if subjects is not None:
            self._epoch_subjects.extend(subjects)

    def on_epoch_end(self, verbose=True, verbose_sub_groups=False) -> None:
        """Updates the metrics, and should be called after each epoch"""
        self._update_metrics()
        if verbose:
            self._print_newest_metrics()
        if verbose_sub_groups and self._subgroup_histories is not None:
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
        if self._subgroup_histories is not None:
            for subgroups in self._subgroup_histories.values():
                # A value in subgroups could e.g. be {"red_bull": {"mse": [val1, val2], "mae": [val3, val4]}}
                for sub_group_name, sub_group_metrics in subgroups.items():
                    for i, (metric_name, metric_values) in enumerate(sub_group_metrics.items()):
                        # With the current implementation, not all subgroups levels passed to __init__ must have been
                        # seen
                        if not metric_values:
                            continue

                        # Print metrics
                        if i == len(self.history) - 1:
                            if self._name is None:
                                print(f"{sub_group_name}_{metric_name}: {metric_values[-1]:.3f}")
                            else:
                                print(f"{self._name}_{sub_group_name}_{metric_name}: {metric_values[-1]:.3f}")
                        else:
                            if self._name is None:
                                print(f"{sub_group_name}_{metric_name}: {metric_values[-1]:.3f}\t\t", end="")
                            else:
                                print(f"{self._name}_{sub_group_name}_{metric_name}: {metric_values[-1]:.3f}\t\t",
                                      end="")

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
        if self._subgroup_histories is not None:
            # Make dictionary containing subjects combined with the prediction and target. The prediction is the average
            # of all EEG epochs/segments
            subjects_predictions: Dict[Subject, List[YYhat]] = dict()
            for subject, y_hat, y in zip(self._epoch_subjects, y_pred, y_true):
                if subject in subjects_predictions:
                    subjects_predictions[subject].append(YYhat(y_true=y, y_pred=y_pred))
                else:
                    subjects_predictions[subject] = [YYhat(y_true=y, y_pred=y_pred)]

            subjects_pred_and_true: Dict[Subject, YYhat] = dict()
            for subject, predictions_and_truths in subjects_predictions.items():
                # Verify that the ground truth is the same
                # todo: consider removing this, as it is not necessarily true in self-supervised learning
                all_ground_truths = set(yyhat.y_true for yyhat in predictions_and_truths)
                if len(all_ground_truths) == 1:
                    raise ValueError(f"Expected all ground truths to be the same per subject, but found "
                                     f"{len(all_ground_truths)} unique ones")

                # Set prediction to the average of all predictions, and the ground truth to the only element in the set
                _pred = torch.mean(torch.cat([torch.unsqueeze(yyhat.y_pred, dim=0)
                                              for yyhat in predictions_and_truths], dim=0), dim=0)
                _true = tuple(all_ground_truths)[0]
                subjects_pred_and_true[subject] = YYhat(y_pred=_pred, y_true=_true)

            # Loop through all splits
            for split_level, sub_groups in self._subgroup_histories.items():
                for sub_group_name, sub_group_metrics in sub_groups.items():
                    # A value of 'sub_group_metrics' could e.g. be {"mse": [val1, val2], "mae": [val3, val4]}

                    # Extract the subgroup
                    sub_group_subjects = tuple(subject for subject in self._epoch_subjects
                                               if subject[split_level] == sub_group_name)

                    # Exit if there are no subject in the subgroup
                    if not sub_group_subjects:
                        continue

                    # Extract their predictions and targets
                    sub_group_y_pred = torch.cat([subjects_pred_and_true[subject].y_pred
                                                  for subject in sub_group_subjects], dim=0)
                    sub_group_y_true = torch.cat([subjects_pred_and_true[subject].y_true
                                                  for subject in sub_group_subjects], dim=0)

                    # Loop through and calculate the metrics
                    for metric_name, metric_values in sub_group_metrics.items():
                        # Compute metrics for the subgroup and store it
                        metric_values.append(self._compute_metric(metric=metric_name, y_pred=sub_group_y_pred,
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

        # Input check
        if metric not in cls.get_available_metrics():
            raise ValueError(f"The metric {metric} was not recognised. The available ones are: "
                             f"{cls.get_available_metrics()}")

        # Compute the metric
        return getattr(cls, metric)(y_pred=y_pred, y_true=y_true)

    # -----------------
    # Properties
    # -----------------
    @property
    def name(self) -> str:
        # todo: I don't like this, looks like a quick fix...
        return "UNNAMED" if self._name is None else self._name  # type: ignore[no-any-return]

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
        # --------------
        # Sanity checks
        # --------------
        # Check if number of epochs is the same for all subjects
        num_epochs = len(tuple(self._prediction_history.values())[0])  # type: ignore
        assert all(len(predictions) == num_epochs for predictions in self._prediction_history.values())

        # Check if number of EEG epochs is the same for all subjects and epochs
        all_num_eeg_epochs = set(len(eeg_epoch_predictions) for epoch_predictions in self._prediction_history.values()
                                 for eeg_epoch_predictions in epoch_predictions)
        assert len(all_num_eeg_epochs) == 1, (f"Expected number of EEG epochs to be the same for all epochs and "
                                              f"subjects, but found {all_num_eeg_epochs}")
        num_eeg_epochs = tuple(all_num_eeg_epochs)[0]

        # --------------
        # Saving of prediction history
        # --------------
        # Flatten out the prediction history
        prediction_history = {sub_id: tuple(itertools.chain(*epoch_predictions))
                              for sub_id, epoch_predictions in self._prediction_history.items()}

        # Create pandas dataframe with the prediction histories
        epochs_column_names = [f"pred{j+1}_epoch{i+1}" for i in range(num_epochs) for j in range(num_eeg_epochs)]
        df = pandas.DataFrame.from_dict(prediction_history, orient="index", columns=epochs_column_names)

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
        df.to_csv(os.path.join(path, f"{history_name}_predictions.csv"), index=False)

    def save_subgroup_metrics(self, history_name, path, decimals=None):
        # If there are no subgroups registered, raise a warning and do nothing
        if self._subgroup_histories is None:
            warnings.warn(message="Tried to save plot of metrics computed per sub-group, but there were no subgroups",
                          category=RuntimeWarning)
            return None

        # Loop through all levels
        for level, subgroups in self._subgroup_histories.items():
            # Get the metrics
            metrics: Tuple[str, ...] = ()
            for subgroup_metrics in subgroups.values():
                if not metrics:
                    metrics = tuple(subgroup_metrics.keys())
                else:
                    # All metrics used should be the same, but different order is ok
                    if not set(metrics) == set(subgroup_metrics.keys()):
                        raise RuntimeError("Expected all metrics to be the same for all sub-groups, but that was not "
                                           "the case")

            # Create folder
            level_path = os.path.join(path, level)
            if not os.path.isdir(level_path):
                os.mkdir(level_path)

            # Loop through and create a plot per metrics (and level)
            for metric_to_plot in metrics:
                pyplot.figure(figsize=(12, 6))  # todo: don't hardcode

                # Make folder
                metric_path = os.path.join(level_path, metric_to_plot)
                if not os.path.isdir(metric_path):
                    os.mkdir(metric_path)

                # Loop through all subgroups
                df_dict: Dict[str, List[float]] = dict()
                for subgroup_name, subgroup_metrics in subgroups.items():
                    # Get the performance
                    performance = subgroup_metrics[metric_to_plot]

                    # Only add it if it is non-empty
                    if performance:
                        df_dict[subgroup_name] = performance

                    # Plot, if values are registered
                    if performance:
                        pyplot.plot(range(1, len(performance) + 1), performance, label=subgroup_name)

                # Plot cosmetics
                font_size = 15

                pyplot.title(f"Performance (level={level})", fontsize=font_size+5)
                pyplot.xlabel("Epoch", fontsize=font_size)
                pyplot.ylabel(metric_to_plot.capitalize(), fontsize=font_size)
                pyplot.tick_params(labelsize=font_size)
                pyplot.legend(fontsize=font_size)
                pyplot.grid()

                # Save figure and close it
                pyplot.savefig(os.path.join(metric_path, f"{history_name}_{metric_to_plot}.png"))
                pyplot.close()

                # Save history object as well
                df = pandas.DataFrame.from_dict(df_dict)

                if decimals is not None:
                    df = df.round(decimals)

                df.to_csv(os.path.join(metric_path, f"{history_name}_{metric_to_plot}.csv"), index=False)

    def save_main_history(self, history_name, path, decimals):
        """Method for saving the main (non subgroup) history"""
        # ---------------
        # Save predictions
        # ---------------
        self.save_prediction_history(history_name=history_name, path=path, decimals=decimals)

        # ---------------
        # Save the metrics in .csv format
        # ---------------
        # Create pandas dataframe
        df = pandas.DataFrame.from_dict(self._history)

        # Maybe set decimals
        if decimals is not None:
            df = df.round(decimals)

        # Save as .csv
        df.to_csv(os.path.join(path, f"{history_name}_metrics.csv"), index=False)

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

    @classmethod
    def get_available_metrics(cls):
        """Get all available metrics"""
        return (cls.get_available_classification_metrics() + cls.get_available_multiclass_classification_metrics() +
                cls.get_available_regression_metrics())

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
        # todo: a value error is raised if only one class is present in y_true
        try:
            return roc_auc_score(y_true=torch.squeeze(y_true, dim=-1).cpu(),
                                 y_score=torch.squeeze(y_pred, dim=-1).cpu())
        except ValueError:
            return numpy.nan

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
    @multiclass_classification_metric
    def auc_ovo(y_pred: torch.Tensor, y_true: torch.Tensor):
        with torch.no_grad():
            return roc_auc_score(y_true=y_true.cpu(), y_score=torch.softmax(y_pred, dim=-1).cpu(), multi_class="ovo")

    @staticmethod
    @multiclass_classification_metric
    def auc_ovr(y_pred: torch.Tensor, y_true: torch.Tensor):
        with torch.no_grad():
            return roc_auc_score(y_true=y_true.cpu(), y_score=torch.softmax(y_pred, dim=-1).cpu(), multi_class="ovr")

    @staticmethod
    @multiclass_classification_metric
    def ce_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
        with torch.no_grad():
            performance = nn.CrossEntropyLoss(reduction='mean')(y_pred, y_true).cpu()
        return float(performance)


class DistanceMetrics:
    """
    Class for computing distances between two distributions

    Examples
    --------
    >>> DistanceMetrics.get_available_distance_metrics()
    ('average_l2_to_centroid', 'centroid_l2')
    >>> my_x1 = torch.tensor([[0, 1], [2, 1], [2, -1], [0, -1]], dtype=torch.float)
    >>> my_x2 = torch.tensor([[1, 0], [3, 0], [3, -2], [1, -2]], dtype=torch.float)
    >>> DistanceMetrics.compute_distance(metric="centroid_l2", x1=my_x1, x2=my_x2)  # doctest: +ELLIPSIS
    1.414...
    >>> DistanceMetrics.compute_distance(metric="average_l2_to_centroid", x1=my_x1, x2=my_x2)  # doctest: +ELLIPSIS
    1.707...
    """

    __slots__ = ()

    @classmethod
    def get_available_distance_metrics(cls):
        """Get all distance metrics available for the class. The distance metric must be a method decorated by
        @distance_metric to be properly registered"""
        # Get all metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a distance metric
            if callable(attribute) and getattr(attribute, "_is_distance_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    @classmethod
    def compute_distance(cls, metric, *, x1, x2):
        """Main method for computing distances"""
        # Input check
        if metric not in cls.get_available_distance_metrics():
            raise ValueError(f"The metric {metric} was not recognised. The available metrics for this class are: "
                             f"{cls.get_available_distance_metrics()}")

        # Return distance
        return getattr(cls, metric)(x1=x1, x2=x2)

    # -------------
    # Distance measures
    # -------------
    @staticmethod
    @distance_metric
    def centroid_l2(x1, x2):
        """Compute the L2 distance between the centroids of the distribution"""
        return numpy.linalg.norm(numpy.array(torch.mean(x1, dim=0) - torch.mean(x2, dim=0)), ord=2)

    @staticmethod
    @distance_metric
    def average_l2_to_centroid(x1, x2):
        """Measures the average distance from the points of x2 to the centroid of x1. This is not actually a valid
        distance metric"""
        return numpy.mean(numpy.linalg.norm(numpy.array(torch.mean(x1, dim=0, keepdim=True) - x2), ord=2, axis=-1))


# ----------------
# Functions
# ----------------
def compute_distribution_distance(metric: str, *, x1: torch.Tensor, x2: torch.Tensor):
    """Function for computing the distance between two distributions"""
    return DistanceMetrics.compute_distance(metric=metric, x1=x1, x2=x2)


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
    _all_metrics = []
    for history in histories:
        _all_metrics.extend(list(history.history.keys()))
    all_metrics = set(_all_metrics)  # Keep unique ones only

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


def save_test_histories_plots(path, histories):
    """
    Save histories

    todo: now we have two...
    Parameters
    ----------
    path : str
    histories : dict[str, Histories]

    Returns
    -------
    None
    """
    # ----------------
    # Loop through all metrics
    # ----------------
    # Get all available metrics
    _all_metrics: Tuple[str, ...] = ()
    for name, history in histories.items():
        if not _all_metrics:
            _all_metrics += tuple(history.history.keys())
        else:
            if set(_all_metrics) != set(history.history.keys()):
                raise RuntimeError("Expected all metrics to be the same, but that was not the case")
    all_metrics = tuple(_all_metrics)

    for metric in all_metrics:
        pyplot.figure(figsize=(12, 6))

        for name, history in histories.items():
            pyplot.plot(range(1, len(history.history[metric]) + 1), history.history[metric], label=name)

        # ------------
        # Plot cosmetics
        # ------------
        font_size = 15

        pyplot.title(f"Test performance ({metric.capitalize()})", fontsize=font_size + 5)
        pyplot.xlabel("Epoch", fontsize=font_size)
        pyplot.ylabel(metric.capitalize(), fontsize=font_size)
        pyplot.tick_params(labelsize=font_size)
        pyplot.legend(fontsize=font_size)
        pyplot.grid()

        # Save figure and close it
        pyplot.savefig(os.path.join(path, f"{metric}.png"))
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
    _all_metrics: Tuple[str, ...] = ()
    if train_history is not None:
        _all_metrics += tuple(train_history.history.keys())
    if val_history is not None:
        _all_metrics += tuple(val_history.history.keys())
    if test_history is not None:
        _all_metrics += tuple(test_history.history.keys())
    if test_estimate is not None:
        _all_metrics += tuple(test_estimate.history.keys())
    all_metrics = set(_all_metrics)

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
                        label="Test estimate", color="red")

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
