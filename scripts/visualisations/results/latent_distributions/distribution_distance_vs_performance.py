"""
This script plots the test set performance vs. the distance from the test dataset to the others. The plot is coloured
with the performance on the validation set.
"""
import os
from typing import List

import matplotlib
import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.paths import get_results_dir


# --------------
# Functions for getting the distribution distance
# --------------
def _get_distances(*, path, metric):
    return pandas.read_csv(os.path.join(path, f"{metric}.csv"), index_col=0)


def _aggregate_distances(*, distances: pandas.DataFrame, method: str):
    # Input check
    assert distances.ndim == 1, (f"Expected input to be 1 dimensional, as the dataset should already have been "
                                 f"selected. Instead, {distances.ndim} number of dimensions were found.")

    # Aggregate
    if method == "mean":
        return numpy.mean(distances[0])
    elif method == "median":
        return numpy.median(distances[0])
    else:
        raise ValueError(f"Aggregation method {method} was not recognised")


def _get_aggregated_distance(*, path, dataset, metric, exclude_self, aggregation_method, scale):
    # Get the distances
    df = _get_distances(path=path, metric=metric)

    # Maybe scale based on diagonal entries
    if scale:
        for row in df.index:
            df.at[row, dataset] /= df.at[row, row]

    # Maybe exclude the self column
    if exclude_self:
        df.drop(dataset, axis="index", inplace=True)

    # Aggregate the distances
    return _aggregate_distances(distances=df[dataset], method=aggregation_method)


# --------------
# Functions for getting performances
# --------------
def _get_val_test_performance(path, *, metric, balance_validation_performance):
    # Input check
    if not isinstance(balance_validation_performance, bool):
        raise TypeError

    # Get the validation and test performances
    if balance_validation_performance:
        val_df_path = os.path.join(path, "sub_groups_plots", "dataset_name", metric, f"val_{metric}.csv")
        val_df = pandas.read_csv(val_df_path)  # type: ignore

        # todo: consider merging by median
        val_performances = numpy.mean(val_df.values, axis=-1)

        # Get the best performance and its epoch  # todo: assumes the higher the better
        val_idx = numpy.argmax(val_performances)
        val_performance = val_performances[val_idx]

    else:
        # Load the dataframe of the validation performances
        val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

        # Get the best performance and its epoch  # todo: assumes the higher the better
        val_idx = numpy.argmax(val_df[metric])
        val_performance = val_df[metric][val_idx]

    # Get the test performance from the same epoch
    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_performance = test_df[metric][val_idx]

    return val_performance, test_performance


# --------------
# Functions for getting the correct path
# --------------
def _select_correct_fold(dataset, run_fold):
    """Function which selects the correct fold (they are currently called 'Fold_0', 'Fold_1' and so on, so we will
    access the test history object to infer it). A test is also made to ensure that the test set only contain one
    dataset"""
    # Get all fold paths
    folds = tuple(path for path in os.listdir(run_fold) if path[:5] == "Fold_")

    # Loop through all folds
    for fold in folds:
        # Load the test predictions
        test_df = pandas.read_csv(os.path.join(run_fold, fold, "test_history_predictions.csv"))

        # Check the number of datasets in the test set
        if len(set(test_df["dataset"])) != 1:
            raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                             f"the case for the fold at {os.path.join(run_fold, fold)}. Found "
                             f"{set(test_df['dataset'])}")

        # If there is a match, return the fold
        if test_df["dataset"][0] == dataset:
            return fold

    # If no match, raise error
    raise ValueError(f"The dataset {dataset} was not found in the directory {run_fold}")


def _get_path(run, results_dir, dataset):
    run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

    # Find out which fold is the correct one for the dataset
    fold = _select_correct_fold(dataset=dataset, run_fold=run_path)

    # Merge to obtain absolute path
    return os.path.join(run_path, fold)  # type: ignore


# --------------
# Main function
# --------------
def main():
    # --------------
    # Hyperparameters
    # --------------
    distance_metric = "average_l2_to_centroid"  # "centroid_l2"
    scale = True
    distance_aggregation_method = "median"
    exclude_self = True

    dataset = "yulin_wang"
    performance_metric = "auc"
    balance_validation_performance = False

    results_dir = os.path.join(get_results_dir(), "debug_plot_script_new")

    # Cosmetics
    colormap = "Blues"
    fontsize = 17
    figsize = (16, 9)

    # --------------
    # Define some prettier names
    # --------------
    pretty_name = {"auc": "AUC",
                   "average_l2_to_centroid": "Average L2 norm to centroid",
                   "centroid_l2": "Centroid L2 distance",
                   "hatlestad_hall": "HatlestadHall",
                   "yulin_wang": "YulinWang",
                   "rockhill": "Rockhill",
                   "mpi_lemon": "MPI Lemon"}

    # --------------
    # Select runs
    # --------------
    runs = os.listdir(results_dir)

    # --------------
    # Get performances and distances per run
    # --------------
    val_performance: List[float] = []
    test_performance: List[float] = []
    distances: List[float] = []
    for run in runs:
        # Get the path
        path = _get_path(run=run, results_dir=results_dir, dataset=dataset)

        # Get performance
        val, test = _get_val_test_performance(path=path, metric=performance_metric,
                                              balance_validation_performance=balance_validation_performance)
        val_performance.append(val)
        test_performance.append(test)

        # Get distance
        distances.append(_get_aggregated_distance(aggregation_method=distance_aggregation_method,
                                                  exclude_self=exclude_self, dataset=dataset, metric=distance_metric,
                                                  path=os.path.join(path, "latent_features"),  # type: ignore
                                                  scale=scale))

    # --------------
    # Plotting
    # --------------
    pyplot.figure(figsize=figsize)

    pyplot.scatter(x=distances, y=val_performance, c=test_performance, cmap=matplotlib.colormaps.get_cmap(colormap),
                   marker="o")

    # Cosmetics
    pyplot.xlabel(pretty_name[distance_metric], fontsize=fontsize)
    pyplot.ylabel(pretty_name[performance_metric], fontsize=fontsize)
    pyplot.xticks(fontsize=fontsize)
    pyplot.yticks(fontsize=fontsize)
    pyplot.title(pretty_name[dataset], fontsize=fontsize)
    pyplot.colorbar()

    pyplot.show()


if __name__ == "__main__":
    main()
