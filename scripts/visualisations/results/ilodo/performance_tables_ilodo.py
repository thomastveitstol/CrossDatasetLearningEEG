import dataclasses
import os
from typing import Dict

import numpy
import pandas

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.analysis.results_analysis import get_all_ilodo_runs, higher_is_better, is_better, get_ilodo_val_dataset_name


# ----------------
# Convenient class
# ----------------
@dataclasses.dataclass(frozen=True)
class _Model:
    path: str  # Absolute path to the results
    train_dataset: str  # The dataset which was used for training
    best_epoch: int  # The best epoch


# ----------------
# Functions for getting the results
# ----------------
def _get_lodo_val_metrics(path, *, main_metric):
    """
    Function for getting the best validation metric for a single fold. It also returns the best epoch and the name of
    the trainig dataset

    Parameters
    ----------
    path : str
    main_metric : str

    Returns
    -------
    tuple[float, int, str]
        Validation performance, the best epoch, and train dataset name
    """
    # --------------
    # Get training dataset name
    # --------------
    dataset_name = get_ilodo_val_dataset_name(path)

    # --------------
    # Get validation performance and optimal epoch
    # --------------
    # Load the dataframe of the validation performances
    val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

    # Get the best performance and its epoch
    if higher_is_better(metric=main_metric):
        best_epoch = numpy.argmax(val_df[main_metric])
    else:
        best_epoch = numpy.argmin(val_df[main_metric])
    val_metric = val_df[main_metric][best_epoch]

    return val_metric, best_epoch, dataset_name


def _get_ilodo_test_metrics(path, epoch):
    # -----------------
    # Get the test metrics per test dataset
    # -----------------
    # Get path to where the metrics are stored
    subgroup_path = os.path.join(path, "sub_groups_plots", "dataset_name")  # hard-coded for now
    metrics = os.listdir(subgroup_path)

    # Get all metrics
    test_metrics = {}
    for metric in metrics:
        df = pandas.read_csv(os.path.join(subgroup_path, metric, f"test_{metric}.csv"))

        # Loop through all the test datasets
        datasets = df.columns
        for dataset in datasets:
            if dataset not in test_metrics:
                test_metrics[dataset] = {}

            # Add the test performance
            test_metrics[dataset][metric] = df[dataset][epoch]

    # -----------------
    # Add the test metrics on the pooled dataset
    # -----------------
    pooled_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_metrics["Pooled"] = {metric: pooled_df[metric][epoch] for metric in pooled_df.columns}

    return test_metrics


def get_best_ilodo_performances(results_dir, *, main_metric, metrics_to_print):
    # Get all runs for inverted LODO
    runs = get_all_ilodo_runs(results_dir)

    # Initialisation
    best_val_metrics: Dict[str, float] = {}
    best_models: Dict[str, _Model] = {}

    # --------------
    # Loop through all experiments for model selection
    # --------------
    for run in runs:
        try:
            run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

            # Get the performances per fold
            folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
            for fold in folds:
                # I need the validation performance and the dataset which was used. The test set performances is not
                # acquired here to reduce runtime
                val_metric, epoch, train_dataset = _get_lodo_val_metrics(
                    path=os.path.join(run_path, fold), main_metric=main_metric  # type: ignore
                )

                # If this is the best run (as evaluated on the validation), store the details
                if train_dataset not in best_val_metrics or is_better(
                        metric=main_metric, old_metrics={main_metric: best_val_metrics[train_dataset]},
                        new_metrics={main_metric: val_metric}
                ):
                    # Update the best model selection
                    best_models[train_dataset] = _Model(path=os.path.join(run_path, fold),  # type: ignore
                                                        train_dataset=train_dataset,
                                                        best_epoch=epoch)

                    # Update best metrics
                    if train_dataset in best_val_metrics:
                        print(f"{train_dataset}: {best_val_metrics[train_dataset]:.2f} < {val_metric:.2f}")
                    best_val_metrics[train_dataset] = val_metric
        except KeyError:
            continue

    # --------------
    # Get the test performances from the best models
    # --------------
    # E.g., {"TDBrain": {"LEMON": {"mse": 10.4, "mae": 2.7}}}
    test_performances: Dict[str, Dict[str, Dict[str, float]]] = {}
    for train_dataset, model in best_models.items():
        test_performances[train_dataset] = _get_ilodo_test_metrics(path=model.path, epoch=model.best_epoch)

    # --------------
    # Print out the results
    # --------------
    print(f"{' Inverted LODO ':=^30}")
    for train_dataset, generalisation_performances in test_performances.items():
        model = best_models[train_dataset]

        print(f"{f' {train_dataset} ':-^20}")
        print(f"Best run: {model.path}")
        for test_dataset, performances in generalisation_performances.items():
            for metric in metrics_to_print:
                print(f"\t{train_dataset} -> {test_dataset} ({metric}): {performances[metric]:.2f}")


def main():
    # Hyperparameters
    main_metric = "pearson_r"
    metrics_to_print = ("pearson_r", "r2_score", "mae")

    # Print results
    get_best_ilodo_performances(results_dir=get_results_dir(), main_metric=main_metric,
                                metrics_to_print=metrics_to_print)


if __name__ == "__main__":
    main()
