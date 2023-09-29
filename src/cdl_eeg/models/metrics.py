"""
Implementing classification Histories class for storing training and validation metrics during training

There will likely be overlap with the RBP implementation at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/metrics.py

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
from typing import Dict, List

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, roc_auc_score
import torch


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

    Examples
    --------
    >>> Histories.get_available_classification_metrics()
    ('auc',)
    >>> Histories.get_available_regression_metrics()
    ('mae', 'mape', 'mse', 'pearson_r', 'spearman_rho')
    """

    __slots__ = "_history", "_epoch_y_pred", "_epoch_y_true", "_name"

    def __init__(self, metrics, name=None):
        """
        Initialise

        Parameters
        ----------
        metrics : str | tuple[str, ...]
            The metrics to use. If 'str', it must either be 'regression' or 'classification', specifying that all
            available regression/classification metrics should be used
        name : str, optional
            May be used for the printing of the metrics
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

        # Create history dictionary
        self._history: Dict[str, List[float]] = {f"{metric}": [] for metric in metrics}

        # Initialise epochs predictions and targets. They will be updated for each batch
        self._epoch_y_pred: List[torch.Tensor] = []
        self._epoch_y_true: List[torch.Tensor] = []

    def store_batch_evaluation(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Store the predictions and targets. Should be called for each batch"""
        self._epoch_y_pred.append(y_pred)
        self._epoch_y_true.append(y_true)

    def on_epoch_end(self, verbose=True) -> None:
        """Updates the metrics, and should be called after each epoch"""
        self._update_metrics()
        if verbose:
            self._print_newest_metrics()

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

    def _update_metrics(self):
        # Update all metrics
        for metric, hist in self._history.items():
            hist.append(self._compute_metric(metric=metric, y_pred=torch.cat(self._epoch_y_pred, dim=0),
                                             y_true=torch.cat(self._epoch_y_true, dim=0)))

        # Remove the epoch histories
        self._epoch_y_pred = []
        self._epoch_y_true = []

    @classmethod
    def _compute_metric(cls, metric, *, y_pred, y_true):
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
