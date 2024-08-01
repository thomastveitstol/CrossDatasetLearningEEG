import os
from typing import Dict, NamedTuple, List

import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.paths import get_results_dir

# ----------------
# Globals
# ----------------
PRETTY_NAME = {"pearson_r": "Pearson's r",
               "spearman_rho": "Spearman's rho",
               "r2_score": r"$R^2$",
               "mae": "MAE",
               "mse": "MSE",
               "HatlestadHall": "SRM",
               "Miltiadous": "Miltiadous",
               "YulinWang": "Wang",
               "MPILemon": "LEMON",
               "TDBrain": "TDBRAIN",
               "Pooled": "Pooled"}


# ----------------
# Smaller convenient functions
# ----------------
def _higher_is_better(metric):
    if metric in ("pearson_r", "spearman_rho", "r2_score"):
        return True
    elif metric in ("mae", "mse", "mape"):
        return False
    else:
        raise ValueError(f"Metric {metric} not recognised")


def _get_all_ilodo_runs(results_dir):
    """Function for getting all runs available for leave-one-dataset-out"""
    return tuple(run for run in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, run,
                                                                                       "leave_one_dataset_out",
                                                                                       "finished_successfully.txt"))
                 and "inverted_cv" in run)


def _get_ilodo_val_dataset_name(path) -> str:
    """Function for getting the name of the validation dataset. A test is also made to ensure that the validation set
    only contains one dataset"""
    # Load the validation predictions
    val_df = pandas.read_csv(os.path.join(path, "val_history_predictions.csv"))

    # Check the number of datasets in the validation set
    if len(set(val_df["dataset"])) != 1:
        raise ValueError(f"Expected only one dataset to be present in the validation set predictions, but that was not "
                         f"the case for the path {path}. Found {set(val_df['dataset'])}")

    # Return the dataset name
    dataset_name: str = val_df["dataset"][0]
    return dataset_name


# ----------------
# Functions for getting the results
# ----------------
def _get_ilodo_test_metrics(path, epoch, datasets):
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

        # Loop through all the test datasets of interest
        for dataset in datasets:
            if dataset not in df.columns:
                # It means that the dataset was used for training
                continue

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


def _get_test_val_metrics(path, *, main_metric, datasets):
    """
    Function for getting the test and validation metric for a single fold. It also returns the best epoch and the name
    of the training dataset

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
    dataset_name = _get_ilodo_val_dataset_name(path)
    if dataset_name not in datasets:
        # If we are not interested, we will not waste time loading data for then to discard the results
        raise _SkipFold

    # --------------
    # Get validation performance and optimal epoch
    # --------------
    # Load the dataframe of the validation performances
    val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

    # Get the best performance and its epoch
    if _higher_is_better(metric=main_metric):
        best_epoch = numpy.argmax(val_df[main_metric])
    else:
        best_epoch = numpy.argmin(val_df[main_metric])
    val_metric = val_df[main_metric][best_epoch]

    # --------------
    # Get test performance
    # --------------
    test_metrics = _get_ilodo_test_metrics(path, epoch=best_epoch, datasets=datasets)
    return val_metric, test_metrics, dataset_name


class ValTestPerformances(NamedTuple):
    val: float
    test: Dict[str, Dict[str, float]]  # e.g., {"LEMON": {"mse": 10.3}}


def plot_test_vs_val_ilodo(results_dir, *, main_metric, metrics_to_plot, datasets):
    # Get all runs for inverted LODO
    runs = _get_all_ilodo_runs(results_dir)

    # Initialisation
    performances: Dict[str, List[ValTestPerformances]] = {dataset: [] for dataset in datasets}

    # --------------
    # Loop through all experiments
    # --------------
    num_runs = len(runs)
    skipped = {dataset: 0 for dataset in datasets}
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(f"Run {i + 1}/{num_runs}")

        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")
        for fold in folds:
            # I need the validation performance and the dataset which was used. The test set performances is not
            # acquired here to reduce runtime
            try:
                val_metric, test_metrics, train_dataset = _get_test_val_metrics(
                    path=os.path.join(run_path, fold), main_metric=main_metric, datasets=datasets
                )
            except _SkipFold:
                continue
            except KeyError:
                # If the prediction model guessed that all subjects have the same age, for all folds, model selection
                # 'fails'. We'll just skip them
                skipped[_get_ilodo_val_dataset_name(os.path.join(run_path, fold))] += 1  # not my best piece of work...
                continue

            # Add the val and test performances
            performances[train_dataset].append(ValTestPerformances(val=val_metric, test=test_metrics))

    print(f"Skipped runs: {skipped}")
    # --------------
    # Plot the results
    # --------------
    for train_dataset, val_test_performances in performances.items():
        # Get the validation performances
        x = [performance.val for performance in val_test_performances]

        # Get test performances
        test_performance = [performance.test for performance in val_test_performances]

        # The test performances must be plotted per metric
        for metric in metrics_to_plot:
            pyplot.figure(figsize=FIGSIZE)

            # Plot the test performance
            for dataset in test_performance[0]:
                y = [performance[dataset][metric] for performance in test_performance]

                pyplot.plot(x, y, ".", label=PRETTY_NAME[dataset])

            # Plot cosmetics
            pyplot.title(f"Source dataset: {PRETTY_NAME[train_dataset]}", fontsize=FONTSIZE + 5)
            pyplot.ylabel(f"Test performance ({PRETTY_NAME[metric]})", fontsize=FONTSIZE)
            pyplot.xlabel(f"Validation performance ({PRETTY_NAME[main_metric]})", fontsize=FONTSIZE)
            pyplot.tick_params(labelsize=FONTSIZE)
            pyplot.xlim(-0.6, 1)
            pyplot.ylim(-0.6, 1)
            pyplot.legend(fontsize=FONTSIZE)
            pyplot.grid()
            pyplot.tight_layout()

    pyplot.show()


FIGSIZE = (7, 5)
FONTSIZE = 16


def main():
    # Hyperparameters
    main_metric = "pearson_r"
    metrics_to_plot = ("pearson_r",)
    datasets = ("TDBrain", "MPILemon", "HatlestadHall", "Miltiadous", "YulinWang")

    # Plot results
    plot_test_vs_val_ilodo(results_dir=get_results_dir(), main_metric=main_metric, metrics_to_plot=metrics_to_plot,
                           datasets=datasets)


class _SkipFold(Exception):
    ...


if __name__ == "__main__":
    main()
