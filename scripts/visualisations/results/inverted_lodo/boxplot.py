"""
Script for making a boxplot containing all performances from a source domain to the different target domain, after
multiple inverted leave-one-dataset-out cross validation
"""
import os
from typing import Dict, List

import numpy
import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.paths import get_results_dir


# --------------
# Functions for getting performances
# --------------
def _get_test_performances(path, metric):
    # Extract the validation performance (will be needed to select the epoch)
    val_df = pandas.read_csv(os.path.join(path, "sub_groups_plots", "dataset_name", metric,  # type: ignore
                                          f"val_{metric}.csv"))

    # There should only bea single dataset in the validation set
    if len(val_df.columns) != 1:
        raise ValueError(f"Expected only one dataset in the validation set, but found {len(val_df.columns)}: "
                         f"{val_df.columns}")

    # Get the best epoch of best performance  # todo: assumes the higher the better
    val_idx = numpy.argmax(val_df.iloc[:, 0])

    # --------------
    # Get the test performance from the same epoch, for all datasets in the test set
    # --------------
    test_df = pandas.read_csv(os.path.join(path,  "sub_groups_plots", "dataset_name", metric,  # type: ignore
                                           f"test_{metric}.csv"))

    # The validation dataset (we ensured length=1 above) should not be in the test set
    if val_df.columns[0] in test_df.columns:
        raise ValueError(f"Expected validation dataset to not not be in the test data, but it was. Validation dataset: "
                         f"{val_df.columns[0]}. Test dataset: {test_df.columns}")

    # Get performances
    test_performances: Dict[str, float] = dict()
    for dataset in test_df.columns:
        test_performances[dataset] = test_df.at[val_idx, dataset]

    return test_performances


# --------------
# Functions for getting the correct path
# --------------
def _select_correct_fold(source_dataset, run_fold):
    """Function which selects the correct fold (they are currently called 'Fold_0', 'Fold_1' and so on, so we will
    access the train history object to infer it). A test is also made to ensure that the train set only contain one
    dataset"""
    # Get all fold paths
    folds = tuple(path for path in os.listdir(run_fold) if path[:5] == "Fold_")

    # Loop through all folds
    for fold in folds:
        # Load the train predictions
        train_df = pandas.read_csv(os.path.join(run_fold, fold, "train_history_predictions.csv"))

        # Check the number of datasets in the train set
        if len(set(train_df["dataset"])) != 1:
            raise ValueError(f"Expected only one dataset to be present in the train set predictions, but that was not "
                             f"the case for the fold at {os.path.join(run_fold, fold)}. Found "
                             f"{set(train_df['dataset'])}")

        # If there is a match, return the fold
        if train_df["dataset"][0] == source_dataset:
            return fold

    # If no match, raise error
    raise ValueError(f"The source dataset {source_dataset} was not found in the directory {run_fold}")


def _get_path(run, results_dir, source_dataset):
    run_path = os.path.join(results_dir, run, "leave_one_dataset_out")  # todo: should probably change the folder name

    # Find out which fold is the correct one for the source dataset
    fold = _select_correct_fold(source_dataset=source_dataset, run_fold=run_path)

    # Merge to obtain absolute path
    return os.path.join(run_path, fold)  # type: ignore


# ----------------
# Main function
# ----------------
def main():
    folder_name = "easter_runs"  # "debug_results_inverted_lodo_1"

    # --------------
    # Hyperparameters
    # --------------
    source_dataset = "mpi_lemon"
    metric = "auc"

    results_dir = os.path.join(get_results_dir(), folder_name)

    # --------------
    # Define some prettier names
    # --------------
    pretty_name = {"auc": "AUC",
                   "hatlestad_hall": "HatlestadHall",
                   "yulin_wang": "YulinWang",
                   "rockhill": "Rockhill",
                   "mpi_lemon": "MPI Lemon",
                   "miltiadous": "Miltiadous"}

    # --------------
    # Select runs
    # --------------
    runs = (run for run in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, run,
                                                                                  "leave_one_dataset_out",
                                                                                  "finished_successfully.txt"))
            and "inverted_cv" in run)

    # --------------
    # Get performances per run
    # --------------
    all_performances: Dict[str, List[float]] = dict()
    for run in runs:
        # Get absolute path to the fold
        path = _get_path(run=run, results_dir=results_dir, source_dataset=source_dataset)

        # Get the performances from the current run
        curr_performances = _get_test_performances(path=path, metric=metric)

        # Add it
        for _dataset, performance in curr_performances.items():
            dataset = pretty_name[_dataset]  # Just a prettier version of the dataset name to get a pretty plot
            if dataset in all_performances:
                all_performances[dataset].append(performance)
            else:
                all_performances[dataset] = [performance]

    # --------------
    # Plotting
    # --------------
    ax = seaborn.boxplot(data=all_performances)

    # Cosmetics
    fontsize = 17
    ax.tick_params(labelsize=fontsize)
    ax.set_title(f"Source dataset: {pretty_name[source_dataset]}", fontsize=fontsize + 3)
    ax.set_ylabel(f"Performance ({pretty_name[metric]})", fontsize=fontsize)
    ax.grid(axis="y")
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    pyplot.show()


if __name__ == "__main__":
    main()
