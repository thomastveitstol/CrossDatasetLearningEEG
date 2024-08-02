"""
Script for printing and creating performance metric tables
"""
import dataclasses
import os
from typing import Dict

import numpy
import pandas

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.results_analysis import is_better, higher_is_better, get_all_lodo_runs, get_lodo_dataset_name


# ----------------
# Convenient class
# ----------------
@dataclasses.dataclass(frozen=True)
class _Model:
    path: str  # Path to the results
    test_dataset: str  # The dataset which was used for testing
    metrics: Dict[str, float]  # The metrics table (e.g., {"mse": 10.7, "mae": 3.3})


# ----------------
# Functions for getting the results
# ----------------
def _get_lodo_val_test_metrics(path, *, main_metric, balance_validation_performance):
    """
    Function for getting the validation and test metrics from a single fold. The epoch is selected based on validation
    set performance

    Parameters
    ----------
    path : str
    main_metric : str
        The metric which determines what is 'best'
    balance_validation_performance : bool
        If True, the 'best' performance is computed as the average on the datasets. Otherwise, it is computed on a pool
        of all the data

    Returns
    -------
    tuple[float, dict[str, float], str]
        Validation and test set metrics, as well as dataset name
    """
    # Input check
    if not isinstance(balance_validation_performance, bool):
        raise TypeError(f"Expected argument 'balance_validation_performance' to be boolean, but found "
                        f"{type(balance_validation_performance)}")

    # Get the best epoch, as evaluated on the validation set
    if balance_validation_performance:
        val_df_path = os.path.join(path, "sub_groups_plots", "dataset_name", main_metric, f"val_{main_metric}.csv")
        val_df = pandas.read_csv(val_df_path)

        val_performances = numpy.mean(val_df.values, axis=-1)

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_performances)
        else:
            val_idx = numpy.argmin(val_performances)

        # Currently, we only actually need the 'main_metric'
        val_metric = val_performances[val_idx]

    else:
        # Load the dataframe of the validation performances
        val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_df[main_metric])
        else:
            val_idx = numpy.argmin(val_df[main_metric])
        val_metric = val_df[main_metric][val_idx]

    # Return the validation and test performances from the same epoch, as well as the name of the test dataset
    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))

    test_metrics = {metric: test_df[metric][val_idx] for metric in test_df.columns}
    return val_metric, test_metrics, get_lodo_dataset_name(path)


def _get_best_lodo_performances(results_dir, *, main_metric, balance_validation_performance):
    # Get all runs for LODO
    runs = get_all_lodo_runs(results_dir)

    # Initialisation
    best_val_metrics: Dict[str, float] = {}
    best_models: Dict[str, _Model] = {}

    # --------------
    # Loop through all experiments
    # --------------
    for run in runs:
        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            # I need to get the validation performance (for making model selection), test performance (in case the
            # current run and fold is best for this dataset), and the dataset name (to know which dataset the
            # metrics are for)
            try:
                val_metric, test_metrics, test_dataset = _get_lodo_val_test_metrics(
                    path=os.path.join(run_path, fold), main_metric=main_metric,  # type: ignore
                    balance_validation_performance=balance_validation_performance
                )
            except KeyError:
                continue

            # If this is the best run (as evaluated on the validation), store it. (Not the nicest implementation but
            # it'll do)
            if test_dataset not in best_val_metrics or is_better(
                    metric=main_metric, old_metrics={main_metric: best_val_metrics[test_dataset]},
                    new_metrics={main_metric: val_metric}
            ):
                # Update the best model selection
                best_models[test_dataset] = _Model(path=run, test_dataset=test_dataset, metrics=test_metrics)

                # Update best metrics
                if test_dataset in best_val_metrics:
                    print(f"{test_dataset}: {best_val_metrics[test_dataset]:.2f} < {val_metric:.2f}")
                best_val_metrics[test_dataset] = val_metric

    # --------------
    # Print results
    # --------------
    print(f"{' Results ':=^30}")
    for dataset, model in best_models.items():
        assert dataset == model.test_dataset

        print(f"{f' {dataset} ':-^20}")
        print(f"Best run: {model.path}")
        for metric, performance in model.metrics.items():
            print(f"\t{metric}: {performance:.2f}")


def main():
    # Hyperparameters
    main_metrics = ("pearson_r", "r2_score", "mae")
    balance_validation_performance = False

    # Print results
    for main_metric in main_metrics:
        print(f"{f' Main metric: {main_metric} ':=^30}")
        _get_best_lodo_performances(results_dir=get_results_dir(), main_metric=main_metric,
                                    balance_validation_performance=balance_validation_performance)


if __name__ == "__main__":
    main()
