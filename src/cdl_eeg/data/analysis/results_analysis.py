"""
Functions for analysing the results
"""
import os

import numpy
import pandas
import yaml  # type: ignore[import]

# ----------------
# Globals
# ----------------
PRETTY_NAME = {
    "pearson_r": "Pearson's r",
    "spearman_rho": "Spearman's rho",
    "r2_score": r"$R^2$",
    "mae": "MAE",
    "mse": "MSE",
    "mape": "MAPE",
    "HatlestadHall": "SRM",
    "Miltiadous": "Miltiadous",
    "YulinWang": "Wang",
    "MPILemon": "LEMON",
    "TDBrain": "TDBRAIN",
    "Pooled": "Pooled",
    "MSELoss": "MSE",
    "L1Loss": "MAE",
    "ShallowFBCSPNetMTS": "ShallowFBCSPNet",
    "Deep4NetMTS": "Deep4Net",
    "spline": "Spline",
    "2 * f max": r"$2 \cdot f_{max}$",
    "4 * f max": r"$4 \cdot f_{max}$",
    "8 * f max": r"$8 \cdot f_{max}$"
}


# ----------------
# Exceptions
# ----------------
class SkipFold(Exception):
    """Can be used if you tried to get the results from a fold, but want to skip it instead"""


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

    # If the new metric is nan, we say that it is not better
    if numpy.isnan(new_metrics[metric]):
        return False

    # If the old metric is nan and the new is not, then it is regarded as an improvement
    if numpy.isnan(old_metrics[metric]) and not numpy.isnan(new_metrics[metric]):
        return True

    # Return
    if higher_is_better(metric=metric):
        return new_metrics[metric] > old_metrics[metric]
    else:
        return new_metrics[metric] < old_metrics[metric]


def get_config_file(results_folder, preprocessing):
    file_name = "preprocessing_config.yml" if preprocessing else "config.yml"
    with open(os.path.join(results_folder, file_name)) as f:
        config = yaml.safe_load(f)
    return config


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
        # I mistakenly quit this run by exiting Pycharm. I think it is best to just exclude it for that reason
        _mistake_exit = "age_cv_experiments_2024-09-30_130737"

        # This was terminated after the first two months of running by KeyboardInterrupt
        _keybord_exit = "age_cv_experiments_2024-07-31_090027"
        return tuple(run for run in os.listdir(results_dir) if "inverted_cv" not in run
                     and run not in (_mistake_exit, _keybord_exit))


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


def get_lodo_dataset_name(path) -> str:
    """Function for getting the name of the test dataset. A test is also made to ensure that the test set only contains
    one dataset"""
    # Load the test predictions
    test_df = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))

    # Check the number of datasets in the test set
    if len(set(test_df["dataset"])) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case for the path {path}. Found {set(test_df['dataset'])}")

    # Return the dataset name
    dataset_name: str = test_df["dataset"][0]
    return dataset_name
