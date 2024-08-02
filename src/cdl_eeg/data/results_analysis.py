"""
Functions for analysing the results
"""
import os

import pandas


# ----------------
# Smaller convenient functions
# ----------------
def higher_is_better(metric):
    if metric in ("pearson_r", "spearman_rho", "r2_score"):
        return True
    elif metric in ("mae", "mse", "mape"):
        return False
    else:
        raise ValueError(f"Metric {metric} not recognised")


def is_better(metric, *, old_metrics, new_metrics):
    # Input checks
    assert isinstance(metric, str), f"Expected metric to be string, but found {type(metric)}"
    assert isinstance(old_metrics, dict), f"Expected 'old_metrics' to be dict, but found {type(old_metrics)}"
    assert isinstance(new_metrics, dict), f"Expected 'new_metrics' to be dict, but found {type(new_metrics)}"

    # If old metrics is and empty dict, the new ones are considered the best
    if not old_metrics:
        return True

    # Return
    if higher_is_better(metric=metric):
        return new_metrics[metric] > old_metrics[metric]
    else:
        return new_metrics[metric] < old_metrics[metric]


# -------------------
# Function for getting run folder names
# -------------------
def get_all_lodo_runs(results_dir, successful_only: bool = True):
    """Function for getting all runs available for leave-one-dataset-out"""
    assert isinstance(successful_only, bool)

    if successful_only:
        return tuple(run for run in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, run,
                                                                                           "leave_one_dataset_out",
                                                                                           "finished_successfully.txt"))
                     and "inverted_cv" not in run)
    else:
        return tuple(run for run in os.listdir(results_dir) if "inverted_cv" not in run)


def get_all_ilodo_runs(results_dir, successful_only: bool = True):
    """Function for getting all runs available for inverted leave-one-dataset-out"""
    assert isinstance(successful_only, bool)

    if successful_only:
        return tuple(run for run in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, run,
                                                                                           "leave_one_dataset_out",
                                                                                           "finished_successfully.txt"))
                     and "inverted_cv" in run)
    else:
        return tuple(run for run in os.listdir(results_dir) if "inverted_cv" in run)


# -------------------
# Functions for getting train/test dataset names for a fold
# -------------------
def get_ilodo_val_dataset_name(path) -> str:
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
