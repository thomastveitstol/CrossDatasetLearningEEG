"""
Implementing classification Histories class for storing training and validation metrics during training

There will likely be overlap with the RBP implementation at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/metrics.py

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
from typing import Dict, List, Tuple, Optional, Any

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, roc_auc_score
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

    __slots__ = "_history", "_splits_histories", "_epoch_y_pred", "_epoch_y_true", "_epoch_subjects", "_name"

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
                split_history = dict()
                for split_selection, criteria in split_selections.items():
                    initial_metrics: Dict[str, List[float]] = {f"{metric}": [] for metric in metrics}
                    split_history[split_selection] = {criterion: initial_metrics for criterion in criteria}

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
                print(f"----- Details for split {i} -----")
                print("Inclusion criteria:")
                # todo: mypy complaining?
                for selection, condition in split[0].items():  # type: ignore
                    print(f"\t{selection.capitalize()} must be in: {condition}")
                    # todo: this is where you left...

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
            # todo: y_pred and y_true to NamedTuple or dataclass?
            subjects_pred_and_true = {subject: {"y_pred": y_hat, "y_true": y} for subject, y_hat, y
                                      in zip(self._epoch_subjects, self._epoch_y_pred, self._epoch_y_true)}

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
                            sub_group_y_pred = torch.cat([subjects_pred_and_true[subject]["y_pred"]
                                                          for subject in sub_group_subjects], dim=0)
                            sub_group_y_true = torch.cat([subjects_pred_and_true[subject]["y_true"]
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
    # Methods for computing performance on sub-groups
    # -----------------
    def store_sub_groups_batch_evaluation(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        raise NotImplementedError

    # -----------------
    # Properties
    # -----------------
    @property
    def history(self):
        # todo: consider returning values as tuples
        return self._history

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

    # Classification metrics
    @staticmethod
    @classification_metric
    def auc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return roc_auc_score(y_true=y_true.cpu(), y_score=y_pred.cpu())
