"""
Functions for analysing the results
"""
import dataclasses
import os
from collections.abc import Iterable
from typing import Dict, Literal, Callable, Tuple, Union, Any

import numpy
import pandas
import torch
import yaml  # type: ignore[import]

from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.models.metrics import Histories

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
INV_FREQUENCY_BANDS = {(1.0, 4.0): "Delta",
                       (4.0, 8.0): "Theta",
                       (8.0, 12.0): "Alpha",
                       (12.0, 30.0): "Beta",
                       (30.0, 45.0): "Gamma",
                       (1.0, 45.0): "All"}


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
# Functions for getting performance scores
# -------------------
def get_lodo_validation_performance(path, *, main_metric, balance_validation_performance):
    """Function for getting the validation performance and the selected epoch"""
    # Input check
    if not isinstance(balance_validation_performance, bool):
        raise TypeError(f"Expected argument 'balance_validation_performance' to be boolean, but found "
                        f"{type(balance_validation_performance)}")

    # Get the best epoch, as evaluated on the validation set
    if balance_validation_performance:
        val_df_path = os.path.join(path, "sub_groups_plots", "dataset_name", main_metric, f"val_{main_metric}.csv")
        val_df = pandas.read_csv(val_df_path)  # type: ignore

        val_performances = numpy.mean(val_df.values, axis=-1)

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_performances)
        else:
            val_idx = numpy.argmin(val_performances)

        # Currently, using the 'main_metric'
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

    return val_metric, val_idx


def get_lodi_validation_performance(path, *, main_metric):
    """
    Function for getting the best validation metric for a single fold. It also returns the best epoch and the name of
    the training dataset

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

    return val_metric, best_epoch


def get_lodo_test_performance(path, *, target_metrics, selection_metric, datasets, balance_validation_performance):
    """
    Function for getting the test score

    Parameters
    ----------
    path : str
    target_metrics : tuple[str, ...] | str
    selection_metric : str
    datasets : tuple[str, ...]
    balance_validation_performance : bool

    Returns
    -------
    tuple[dict[str, float], str]
        Test scores and the name of the test dataset
    """
    # --------------
    # Get test dataset name
    # --------------
    dataset_name = get_lodo_dataset_name(path)
    if dataset_name not in datasets:
        # If we are not interested, we will not waste time loading data for then to discard the results
        raise SkipFold

    # --------------
    # Get validation performance and optimal epoch
    # --------------
    val_performance, epoch = get_lodo_validation_performance(
        path=path, main_metric=selection_metric, balance_validation_performance=balance_validation_performance
    )

    # --------------
    # Get test performance
    # --------------
    target_metrics = (target_metrics,) if isinstance(target_metrics, str) else target_metrics

    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_performance = {target_metric: test_df[target_metric][epoch] for target_metric in target_metrics}

    return test_performance, dataset_name, val_performance


def get_lodi_test_performance(path, *, target_metrics, selection_metric, datasets, refit_intercept):
    """
    Get the LODI test performance

    Parameters
    ----------
    path : str
    target_metrics : tuple[str, ...]
    selection_metric : str
    datasets : tuple[str, ...]
    refit_intercept : bool
        todo: should be recognised by the metric names itself

    Returns
    -------
    tuple[dict[str, [str, float]], str]
        Performance per dataset, per metric. And the name of the training dataset
    """
    # --------------
    # Get training dataset name
    # --------------
    train_dataset_name = get_ilodo_val_dataset_name(path)
    if train_dataset_name not in datasets:
        # If we are not interested, we will not waste time loading data for then to discard the results
        raise SkipFold

    # --------------
    # Get optimal epoch (as evaluated on the validation set)
    # --------------
    val_performance, best_epoch = get_lodi_validation_performance(path=path, main_metric=selection_metric)

    # -----------------
    # Get the test metrics per test dataset
    # -----------------
    # Get path to where the metrics are stored
    subgroup_path = os.path.join(path, "sub_groups_plots", "dataset_name")  # hard-coded for now
    metrics = os.listdir(subgroup_path) if target_metrics is None else target_metrics

    # Get all metrics
    if refit_intercept:
        test_metrics = _get_lodi_refit_scores(path, best_epoch, metrics)
    else:
        test_metrics = {}
        for metric in metrics:
            df = pandas.read_csv(os.path.join(subgroup_path, metric, f"test_{metric}.csv"))

            # Loop through all the test datasets
            datasets = df.columns
            for dataset in datasets:
                if dataset not in test_metrics:
                    test_metrics[dataset] = {}

                # Add the test performance
                test_metrics[dataset][metric] = df[dataset][best_epoch]

        # Add the test metrics on the pooled dataset
        pooled_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
        test_metrics["Pooled"] = {metric: pooled_df[metric][best_epoch] for metric in metrics}  # pooled_df.columns

    return test_metrics, train_dataset_name, val_performance


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


# ----------------
# Functions for getting HP configurations
# ----------------
@dataclasses.dataclass(frozen=True)
class HP:
    config_file: Literal["normal", "preprocessing"]  # Indicates which config file to open to find the HPC
    # How to find the HPC in the config file
    location: Union[Tuple[str, ...], Callable]  # type: ignore[type-arg]


def _get_band_pass_filter(config):
    l_freq, h_freq = config["general"]["filtering"]
    return INV_FREQUENCY_BANDS[(l_freq, h_freq)]


HYPERPARAMETERS = {
    "DL architecture": HP(config_file="normal", location=("DL Architecture", "model")),
    "Band-pass filter": HP(config_file="preprocessing", location=_get_band_pass_filter)
}


def _get_hyperparameter_configuration(config_files: Dict[Literal["normal", "preprocessing"], Dict[str, Any]], hp: HP):
    if isinstance(hp.location, Iterable):
        # Traverse to the HPC
        hpc_traversed = config_files[hp.config_file]
        for key in hp.location:
            hpc_traversed = hpc_traversed[key]
        return hpc_traversed
    elif callable(hp.location):
        return hp.location(config_files[hp.config_file])
    raise TypeError(f"Unexpected type of HP location (how to find the HPC in a config file): {type(hp.location)}")


def _load_config_files(*, normal_config_needed, preprocessing_config_needed, results_dir,
                       run) -> Dict[Literal["normal", "preprocessing"], Any]:
    # Load yaml files
    config_files: Dict[Literal["normal", "preprocessing"], Any] = dict()  # type: ignore
    if normal_config_needed:
        with open(results_dir / run / "config.yml") as file:
            config_files["normal"] = yaml.safe_load(file)

    if preprocessing_config_needed:
        with open(results_dir / run / "preprocessing_config.yml") as file:
            config_files["preprocessing"] = yaml.safe_load(file)

    # Make sure at least one was added, and return as tuple
    assert config_files
    return config_files


def add_hp_configurations_to_dataframe(df, hps, results_dir, skip_if_exists=True):
    """
    Function for adding hyperparameter configurations to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
         The input dataframe needs the 'run' column to know which run to extract the HPC from
    hps : tuple[str, ...]
        Hyperparameter configurations to add to the dataframe
    results_dir : patlib.Path
    skip_if_exists : bool
        To skip the HP if it already exists in the dataframe

    Returns
    -------
    pandas.DataFrame
    """
    # (Maybe) skipping HPs which are already in the dataframe
    if skip_if_exists:
        hps = tuple(hp for hp in hps if hp not in df.columns)

    # --------------
    # Load the HPCs
    # --------------
    # Check which config files needs to be loaded
    normal_config_needed = any(HYPERPARAMETERS[hp_name].config_file == "normal" for hp_name in hps)
    preprocessing_config_needed = any(HYPERPARAMETERS[hp_name].config_file == "preprocessing" for hp_name in hps)

    # Loop through all unique runs
    runs = set(df["run"])
    hpcs: Dict[str, Any] = {"run": [], **{hp_name: [] for hp_name in hps}}
    try:
        from progressbar import progressbar  # type: ignore
        loop = progressbar(runs, redirect_stdout=True, prefix="Run ")
    except ImportError:
        loop = runs

    for run in loop:
        # Load the config files
        config_files = _load_config_files(
            normal_config_needed=normal_config_needed,  preprocessing_config_needed=preprocessing_config_needed,
            results_dir=results_dir, run=run
        )

        # Get the HPCs
        for hp_name in hps:
            hpcs[hp_name].append(
                _get_hyperparameter_configuration(config_files=config_files, hp=HYPERPARAMETERS[hp_name])
            )

        hpcs["run"].append(run)

    # --------------
    # Add to the dataframe
    # --------------
    return df.merge(pandas.DataFrame(hpcs), on="run")


# ----------------
# Functions for refitting intercept
# ----------------
def _estimate_intercept(df: pandas.DataFrame):
    # Intercept is calculated as sum(y_i - x_i) / n
    return (df["ground_truth"] - df["pred"]).mean()


def get_average_prediction(df: pandas.DataFrame, epoch):
    new_df = {"dataset": df["dataset"], "sub_id": df["sub_id"],
              "pred": df.iloc[:, (5 * epoch + 2):(5 * epoch + 7)].mean(axis=1)}
    return pandas.DataFrame.from_dict(data=new_df)


def _get_lodi_refit_scores(path, epoch, metrics) -> Dict[str, Dict[str, float]]:
    test_predictions = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))
    test_metrics: Dict[str, Dict[str, float]] = dict()

    # ------------
    # Loop through all datasets
    # ------------
    datasets = set(test_predictions["dataset"])
    for dataset in datasets:
        # Get the predictions for this dataset only
        df = test_predictions[test_predictions["dataset"] == dataset].copy()

        # Average the predictions per EEG epoch
        df = get_average_prediction(df=df, epoch=epoch)

        # Add the targets (age)
        target = "age"  # quick-fixed hard coding
        df["ground_truth"] = get_dataset(dataset).load_targets(target=target, subject_ids=df["sub_id"])

        # Estimate the intercept
        new_intercept = _estimate_intercept(df=df)
        df["adjusted_pred"] = df["pred"] + new_intercept

        # Add the performance
        test_metrics[dataset] = dict()
        for metric in metrics:
            # Normally, I'd add a 'compute_metric' method to Histories, but I don't like to change the code too much
            # after getting the results from a scientific paper, even when it makes sense. So, violating some best
            # practice instead
            test_metrics[dataset][metric] = Histories._compute_metric(
                metric=metric, y_pred=torch.tensor(df["adjusted_pred"].to_numpy()),
                y_true=torch.tensor(df["ground_truth"].to_numpy())
            )

    # ------------
    # Now, fix the pooled dataset
    # ------------
    df = test_predictions.copy()

    # Average the predictions per EEG epoch
    df = get_average_prediction(df=df, epoch=epoch)

    # Sorting makes life easier
    df.sort_values(by=["dataset"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Add the targets (age)
    target = "age"  # quick-fixed hard coding
    ground_truths = []
    for dataset in sorted(datasets):  # Needs to be sorted as in the df
        ground_truths.append(
            get_dataset(dataset).load_targets(target=target, subject_ids=df["sub_id"][df["dataset"] == dataset])
        )
    df["ground_truth"] = numpy.concatenate(ground_truths)
    new_intercept = _estimate_intercept(df=df)
    df["adjusted_pred"] = df["pred"] + new_intercept

    # Add the performance
    test_metrics["Pooled"] = dict()
    for metric in metrics:
        test_metrics["Pooled"][metric] = Histories._compute_metric(
            metric=metric, y_pred=torch.tensor(df["adjusted_pred"].to_numpy()),
            y_true=torch.tensor(df["ground_truth"].to_numpy())
        )
    return test_metrics


def _get_lodi_test_metrics(path, epoch, refit_intercept):
    # -----------------
    # Get the test metrics per test dataset
    # -----------------
    # Get path to where the metrics are stored
    subgroup_path = os.path.join(path, "sub_groups_plots", "dataset_name")  # hard-coded for now
    metrics = os.listdir(subgroup_path)

    # Get all metrics
    if refit_intercept:
        test_metrics = _get_lodi_refit_scores(path, epoch, metrics)
    else:
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

        # Add the test metrics on the pooled dataset
        pooled_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
        test_metrics["Pooled"] = {metric: pooled_df[metric][epoch] for metric in pooled_df.columns}

    return test_metrics
