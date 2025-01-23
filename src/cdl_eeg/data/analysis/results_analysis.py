"""
Functions for analysing the results
"""
import dataclasses
import os
from collections.abc import Iterable
from typing import Dict, Literal, Callable, Tuple, Union, Any, Optional, List, Set

import numpy
import pandas
import torch
import yaml  # type: ignore[import]
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    OrdinalHyperparameter
from ConfigSpace.hyperparameters import Hyperparameter

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
    "mae_refit": r"$MAE_{refit}$",
    "mse_refit": r"$MSE_{refit}$",
    "r2_score_refit": r"$R^2_{refit}$",
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
INV_PRETTY_NAME = {pretty: ugly for ugly, pretty in PRETTY_NAME.items()}
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
def get_label_orders(renamed_df_mapping: Optional[Dict[str, Dict[str, str]]] = None):
    """Function for getting the default order of the labels when making plots"""
    orders: Dict[str, Tuple[str, ...]] = {
        "Target dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang", "Pooled"),
        "Source dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang", "Pooled"),
        "Band-pass filter": ("All", "Delta", "Theta", "Alpha", "Beta", "Gamma"),
        "DL architecture": ("InceptionNetwork", "Deep4NetMTS", "ShallowFBCSPNetMTS"),
        "Normalisation": ("True", "False")
    }
    if renamed_df_mapping is None:
        return orders

    for column, mapping in renamed_df_mapping.items():
        if column in orders:
            updated_order = []
            for value in orders[column]:
                if value in mapping:
                    updated_order.append(mapping[value])
                else:
                    updated_order.append(value)
            orders[column] = tuple(updated_order)
    return orders


def get_formats():
    """Function for getting the default formats when making plots heatmaps"""
    return {"pearson_r": ".2f", "mae": ".1f", "r2_score": ".2f", "r2_score_refit": ".2f"}


def get_rename_mapping():
    """Function for getting a renaming of the labels when making plots"""
    return {"DL architecture": {"InceptionNetwork": "IN", "Deep4NetMTS": "DN", "ShallowFBCSPNetMTS": "SN"}}


def combine_conditions(df: pandas.DataFrame, conditions):
    # Input check
    if not conditions:
        raise RuntimeError("Tried to combine conditions, but there were none passed")

    # Combine conditions
    combined_conditions = True
    for column, category in conditions.items():
        if isinstance(category, (tuple, list, set)):
            combined_conditions &= df[column].isin(category)
        else:  # Try the '==' operator
            combined_conditions &= df[column] == category
    return combined_conditions


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


def get_lodo_test_performance(path, *, target_metrics, selection_metric, datasets, balance_validation_performance,
                              refit_metrics):
    """
    Function for getting the test score

    Parameters
    ----------
    path : str
    target_metrics : tuple[str, ...] | str
    selection_metric : str
    datasets : tuple[str, ...]
    balance_validation_performance : bool
    refit_metrics : tuple[str, ...]
        Same as for LODI

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
    try:
        val_performance, epoch = get_lodo_validation_performance(
            path=path, main_metric=selection_metric, balance_validation_performance=balance_validation_performance
        )
    except KeyError:
        # If the prediction model guessed that all subjects have the same age, for all folds, model selection
        # 'fails'. We'll verify that the selection metric is a correlation metric, set the performance to zero, and the
        # 'best epoch' to the last epoch
        _corr_metrics = ("pearson_r", "spearman_rho")
        assert selection_metric in ("pearson_r", "spearman_rho"), \
            (f"Model selection failed with a selection metric that is not a registered correlation metric. The failed "
             f"metric was {selection_metric}, the registered correlation metrics are {_corr_metrics}")
        epoch = 99  # Last epoch. Got KeyError when using -1, don't know why
        val_performance = 0

    # --------------
    # Get test performance
    # --------------
    target_metrics = (target_metrics,) if isinstance(target_metrics, str) else target_metrics

    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_performance = {target_metric: test_df[target_metric][epoch] for target_metric in target_metrics}

    # Get refit performance scores and add to results
    refit_performance = _get_lodo_refit_scores(path=path, epoch=epoch, metrics=refit_metrics)
    for metric, score in refit_performance.items():
        test_performance[f"{metric}_refit"] = score

    return test_performance, dataset_name, val_performance


def get_lodi_test_performance(path, *, target_metrics, selection_metric, datasets, refit_metrics):
    """
    Get the LODI test performance

    Parameters
    ----------
    path : str
    target_metrics : tuple[str, ...]
    selection_metric : str
    datasets : tuple[str, ...]
    refit_metrics : tuple[str, ...]
        Metrics which you want to refit the intercept for prior to computing them as well

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
    try:
        val_performance, best_epoch = get_lodi_validation_performance(path=path, main_metric=selection_metric)
    except KeyError:
        # If the prediction model guessed that all subjects have the same age, for all folds, model selection
        # 'fails'. We'll verify that the selection metric is a correlation metric, set the performance to zero, and the
        # 'best epoch' to the last epoch
        _corr_metrics = ("pearson_r", "spearman_rho")
        assert selection_metric in ("pearson_r", "spearman_rho"), \
            (f"Model selection failed with a selection metric that is not a registered correlation metric. The failed "
             f"metric was {selection_metric}, the registered correlation metrics are {_corr_metrics}")
        best_epoch = 99  # Last epoch. Got KeyError when using -1, don't know why
        val_performance = 0

    # -----------------
    # Get the test metrics per test dataset
    # -----------------
    # Get path to where the metrics are stored
    subgroup_path = os.path.join(path, "sub_groups_plots", "dataset_name")  # hard-coded for now
    metrics = os.listdir(subgroup_path) if target_metrics is None else target_metrics

    # Get all metrics
    test_metrics: Dict[str, Dict[str, float]] = {}
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

    # Get refit performance scores
    refit_test_metrics = _get_lodi_refit_scores(path, best_epoch, refit_metrics)

    for dataset_name, performances in refit_test_metrics.items():
        for metric, performance in performances.items():
            test_metrics[dataset_name][f"{metric}_refit"] = performance

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
    # The HP distribution or where to find it
    distribution: Union[Hyperparameter, Tuple[str, ...]]  # type: ignore[type-arg]


def _get_band_pass_filter(config):
    l_freq, h_freq = config["general"]["filtering"]
    return INV_FREQUENCY_BANDS[(l_freq, h_freq)]


def _get_normalisation(config):
    if config["Varied Numbers of Channels"]["name"] == "Interpolation":
        return str(config["DL Architecture"]["normalise"])
    elif config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling":
        return str(config["Varied Numbers of Channels"]["kwargs"]["normalise_region_representations"])
    else:
        raise ValueError


# ----------------
# Functions for fANOVA
# ----------------
def _config_dist_to_fanova_dist(distribution, hp_name):
    """Function which maps a distribution as specified in the config file to fANOVA distribution"""
    # Not a very elegant function... should preferably use the sampling distribution functions
    dist_name = distribution["dist"]
    if dist_name == "uniform":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return UniformFloatHyperparameter(name=hp_name, lower=low, upper=high, log=False)
    elif dist_name == "log_uniform":
        base = distribution["kwargs"]["base"]
        low = base ** distribution["kwargs"]["a"]
        high = base ** distribution["kwargs"]["b"]
        return UniformFloatHyperparameter(name=hp_name, lower=low, upper=high, log=True)
    if dist_name == "uniform_int":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return UniformIntegerHyperparameter(name=hp_name, lower=low, upper=high, log=False)
    elif dist_name == "log_uniform_int":
        base = distribution["kwargs"]["base"]
        low = base ** distribution["kwargs"]["a"]
        high = base ** distribution["kwargs"]["b"]
        return UniformIntegerHyperparameter(name=hp_name, lower=low, upper=high, log=True)
    elif dist_name == "n_log_uniform_int":
        # I'll register it as a log scale
        n = distribution["kwargs"]["n"]
        base = distribution["kwargs"]["base"]
        low = n * round(base ** distribution["kwargs"]["a"])
        high = n * round(base ** distribution["kwargs"]["b"])
        return UniformIntegerHyperparameter(name=hp_name, lower=low, upper=high, log=True)
    else:
        raise ValueError(f"Unrecognised distribution: {dist_name}")


def _get_single_hp_distribution(hp, hp_name, hpd_config):
    """
    Get the distribution of an HP, with specified location and the HP distribution config file

    Parameters
    ----------
    hp : HP
    hp_name : str
    hpd_config : dict[str, Any]

    Returns
    -------
    Hyperparameter
    """
    if isinstance(hp.distribution, Hyperparameter):
        return hp.distribution

    hyperparameter_distribution = hpd_config.copy()
    for key in hp.distribution:
        hyperparameter_distribution = hyperparameter_distribution[key]
    return _config_dist_to_fanova_dist(distribution=hyperparameter_distribution, hp_name=hp_name)


def get_fanova_hp_distributions(hp_names, hpd_config):
    """
    Function for getting the fANOVA distributions to be passed to ConfigurationSpace

    Parameters
    ----------
    hp_names: Iterable[str]
    hpd_config : dict[str, Any]
        The configuration file which contains the HP distributions
    Returns
    -------
    dict[str, Hyperparameter]
    """
    hp_distributions: Dict[str, Hyperparameter] = dict()  # type: ignore[type-arg]
    for hp_name in hp_names:
        # Get the HP info
        hp = HYPERPARAMETERS[hp_name]

        # Get the distribution
        hp_distributions[hp_name] = _get_single_hp_distribution(hp=hp, hp_name=hp_name, hpd_config=hpd_config)

    return hp_distributions


def get_fanova_encoding():
    return {
        "RBP": 0, "spline": 1, "MNE": 2,  # Spatial dim
        "All": 0, "Delta": 1, "Theta": 2, "Alpha": 3, "Beta": 4, "Gamma": 5,  # Filters
        "InceptionNetwork": 0, "Deep4NetMTS": 1, "ShallowFBCSPNetMTS": 2,  # Architecture
        "IN": 0, "DN": 1, "SN": 2,  # Architecture acronyms
        "2 * f max": 0, "4 * f max": 1, "8 * f max": 2,  # Sampling frequency
        "False": 0, "True": 1,  # Autoreject and normalisation
        "5s": 0, "10s": 1,  # Input length
        "L1Loss": 0, "MSELoss": 1,  # Loss
        "Nothing": 0, "DD": 1, "CMMN": 2, "CMMN + DD": 3,  # Domain adaptation
    }


def _get_spatial_method(config):
    method = config["Varied Numbers of Channels"]["name"]
    if method == "RegionBasedPooling":
        return "RBP"
    elif method == "Interpolation":
        return config["Varied Numbers of Channels"]["kwargs"]["method"]
    else:
        raise ValueError(f"Unexpected method for handling varied numbers of channels: {method}")


def _get_cmmn(config):
    # How to extract CMMN depends on the method for handling the different electrode configurations
    if config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling":
        use_cmmn: Set[bool] = set()
        for rbp_design in config["Varied Numbers of Channels"]["kwargs"]["RBPDesigns"].values():
            use_cmmn.add(rbp_design["use_cmmn_layer"])

        if len(use_cmmn) != 1:
            raise ValueError(f"Expected all or none of the RBP layers to use CMMN, but found {use_cmmn}")

        # Return the only element in the set
        return tuple(use_cmmn)[0]

    elif config["Varied Numbers of Channels"]["name"] == "Interpolation":
        return config["DL Architecture"]["CMMN"]["use_cmmn_layer"]
    else:
        raise ValueError(f"Unexpected method for handling different electrode configurations: "
                         f"{config['Varied Numbers of Channels']['name']}")


def _get_domain_adaptation(config):
    # Extracting CMMN
    use_cmmn = _get_cmmn(config)

    # Extract use of domain discriminator
    use_domain_discriminator = False if config["DomainDiscriminator"] is None else True

    # Combine outputs and return
    if use_cmmn and use_domain_discriminator:
        return "CMMN + DD"
    elif use_cmmn:
        return "CMMN"
    elif use_domain_discriminator:
        return "DD"
    else:
        return "Nothing"


def _get_sampling_freq(config):
    return f"{int(config['general']['resample'] // config['general']['filtering'][-1])} * f max"


def _get_input_length(config):
    return f"{int(config['general']['num_time_steps'] // config['general']['resample'])}s"


def _get_weighted_loss(config):
    loss_config = config["Training"]["Loss"]  # tau in the paper (as per now, at least)

    # Not using weighted loss is equivalent to tau = 0
    return 0 if loss_config["weighter"] is None else loss_config["weighter_kwargs"]["weight_power"]


CHP = CategoricalHyperparameter
OHP = OrdinalHyperparameter
UFHP = UniformFloatHyperparameter
HYPERPARAMETERS = {
    "DL architecture": HP(config_file="normal", location=("DL Architecture", "model"),
                          distribution=CHP(name="DL architecture", choices=("IN", "DN", "SN"))),
    "Band-pass filter": HP(config_file="preprocessing", location=_get_band_pass_filter,
                           distribution=CHP(name="Band-pass filter", choices=get_label_orders()["Band-pass filter"])),
    "Normalisation": HP(config_file="normal", location=_get_normalisation,
                        distribution=CHP(name="Normalisation", choices=(False, True))),
    "Learning rate": HP(config_file="normal", location=("Training", "learning_rate"),
                        distribution=("Training", "learning_rate")),
    r"$\beta_1$": HP(config_file="normal", location=("Training", "beta_1"),
                     distribution=("Training", "beta_1")),
    r"$\beta_2$": HP(config_file="normal", location=("Training", "beta_2"),
                         distribution=("Training", "beta_2")),
    r"$\epsilon$": HP(config_file="normal", location=("Training", "eps"), distribution=("Training", "eps")),
    "Spatial method": HP(config_file="normal", location=_get_spatial_method,
                         distribution=CHP(name="Spatial method", choices=("spline", "MNE", "RBP"),
                                          weights=(0.25, 0.25, 0.5))),
    "Domain Adaptation": HP(config_file="normal", location=_get_domain_adaptation, distribution=NotImplemented),
    "Sampling frequency": HP(config_file="preprocessing", location=_get_sampling_freq,
                             distribution=OHP(name="Sampling frequency",
                                              sequence=("2 * f max", "4 * f max", "8 * f max"))),
    "Input length": HP(config_file="preprocessing", location=_get_input_length,
                       distribution=OHP(name="Input length", sequence=("5s", "10s"))),
    "Autoreject": HP(config_file="preprocessing", location=("general", "autoreject"),
                     distribution=CHP(name="Autoreject", choices=(False, True))),
    "Loss": HP(config_file="normal", location=("Training", "Loss", "loss"),
               distribution=CHP(name="Loss", choices=("L1Loss", "MSELoss"))),
    r"Weighted loss ($\tau$)": HP(config_file="normal", location=_get_weighted_loss,
                                  distribution=UFHP(name=r"Weighted loss ($\tau$)", lower=0, upper=1))
    # On the weighted loss: Looks to me like only the upper and lower bounds are used anyway (all instances of type
    # NumericalHyperparameter seem to be treated equally). Changing to a normal distribution doesn't change the outcome
    # neither
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


def _get_all_and_average_age(datasets):
    # Get age of all subjects in all datasets
    dataset_age = dict()
    for dataset_name in datasets:
        dataset = get_dataset(dataset_name=dataset_name)
        subject_ids = dataset.get_subject_ids()

        # Add ages (but remove nan values first)
        age_array = dataset.load_targets(target="age", subject_ids=subject_ids)
        dataset_age[dataset_name] = age_array[~numpy.isnan(age_array)]

    # ------------
    # Make all desired combinations
    # ------------
    age_averages: Dict[Union[str, Tuple[str, ...]], float] = dict()
    # LODO
    for dataset_name in dataset_age:
        all_but_current_ages = {d_name: age_vals for d_name, age_vals in dataset_age.items() if d_name != dataset_name}
        dataset_combination = tuple(all_but_current_ages)
        dataset_combination_avg = numpy.mean(numpy.concatenate(tuple(all_but_current_ages.values())))

        # Add
        age_averages[dataset_combination] = dataset_combination_avg  # type: ignore

    # LODI
    for dataset_name, age_values in dataset_age.items():
        age_averages[dataset_name] = numpy.mean(age_values)

    return dataset_age, age_averages


def get_dummy_performance(datasets, metrics):
    dataset_age, age_average = _get_all_and_average_age(datasets=datasets)

    # ------------
    # Compute dummy performance scores
    # ------------
    dummy_performance: Dict[str, List[Union[str, float]]] = {
        "Target dataset": [], "Source dataset": [],
        **{f"Performance ({PRETTY_NAME[metric]})": [] for metric in metrics}
    }

    # LODO
    for target_dataset, ground_truth in dataset_age.items():
        # Make the dummy guess
        guess = numpy.mean(
            numpy.concatenate([age_vals for d_name, age_vals in dataset_age.items() if d_name != target_dataset])
        )

        # Compute dummy performance
        dummy_performance["Source dataset"].append("Pooled")
        dummy_performance["Target dataset"].append(PRETTY_NAME[target_dataset])
        for metric in metrics:
            if metric in ("pearson_r", "spearman_rho"):
                dummy_score = 0
            else:
                dummy_score = Histories._compute_metric(
                    metric=metric, y_pred=torch.tensor([guess] * ground_truth.shape[0]),
                    y_true=torch.tensor(ground_truth)
                )
            dummy_performance[f"Performance ({PRETTY_NAME[metric]})"].append(dummy_score)

    # LODI (single source dataset to single target dataset)
    for source_dataset in dataset_age:
        for target_dataset, ground_truth in dataset_age.items():
            if source_dataset == target_dataset:
                continue

            # Add to dict
            dummy_performance["Source dataset"].append(PRETTY_NAME[source_dataset])
            dummy_performance["Target dataset"].append(PRETTY_NAME[target_dataset])

            guess = age_average[source_dataset]
            for metric in metrics:
                if metric in ("pearson_r", "spearman_rho"):
                    dummy_score = 0
                else:
                    dummy_score = Histories._compute_metric(
                        metric=metric, y_pred=torch.tensor([guess] * ground_truth.shape[0]),
                        y_true=torch.tensor(ground_truth)
                    )
                dummy_performance[f"Performance ({PRETTY_NAME[metric]})"].append(dummy_score)

    # LODI (single source dataset to pooled target dataset)
    for source_dataset in dataset_age:
        ground_truth = numpy.concatenate([ages for dataset, ages in dataset_age.items() if dataset != source_dataset])
        guess = numpy.mean(age_average[source_dataset])

        dummy_performance["Source dataset"].append(PRETTY_NAME[source_dataset])
        dummy_performance["Target dataset"].append("Pooled")
        for metric in metrics:
            if metric in ("pearson_r", "spearman_rho"):
                dummy_score = 0
            else:
                dummy_score = Histories._compute_metric(
                    metric=metric, y_pred=torch.tensor([guess] * ground_truth.shape[0]),
                    y_true=torch.tensor(ground_truth)
                )
            dummy_performance[f"Performance ({PRETTY_NAME[metric]})"].append(dummy_score)

    return pandas.DataFrame(dummy_performance)


def add_hp_configurations_to_dataframe(df, hps, results_dir, skip_if_exists=True):
    """
    Function for adding hyperparameter configurations to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
         The input dataframe needs the 'run' column to know which run to extract the HPC from
    hps : tuple[str, ...]
        Hyperparameter configurations to add to the dataframe
    results_dir : pathlib.Path
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


def extract_selected_best_scores(df, *, selection_metric, target_metrics, target_datasets, source_datasets,
                                 include_std=False, additional_columns=()):
    """Function for getting the selected best test results"""
    # ------------
    # Get the best performing models only
    # ------------
    test_scores: Dict[str, List[Any]] = {"Target dataset": [], "Source dataset": [], "Run": [],
                                         **{f"Performance ({PRETTY_NAME[metric]})": [] for metric in target_metrics},
                                         **{column: [] for column in additional_columns}}
    if include_std:
        test_scores = {**test_scores, **{f"Std ({PRETTY_NAME[metric]})": [] for metric in target_metrics}}

    for source_dataset in source_datasets:
        # Loop through all target datasets
        for target_dataset in target_datasets:
            if target_dataset == source_dataset:
                continue

            # Select the row based on validation performance
            subset_cond = (df["Source dataset"] == source_dataset) & (df["Target dataset"] == target_dataset)
            subset_results_df = df[subset_cond]
            if higher_is_better(selection_metric):
                best_val_run = subset_results_df["run"].loc[subset_results_df["Val score"].idxmax()]
            else:
                best_val_run = subset_results_df["run"].loc[subset_results_df["Val score"].idxmin()]

            # Add the results
            test_scores["Target dataset"].append(target_dataset)
            test_scores["Source dataset"].append(source_dataset)
            test_scores["Run"].append(best_val_run)
            for target_metric in target_metrics:
                # Compute source to target score
                score = subset_results_df[subset_results_df["run"] == best_val_run][target_metric].iloc[0]

                # Add it
                test_scores[f"Performance ({PRETTY_NAME[target_metric]})"].append(score)
                if include_std:
                    test_scores[f"Std ({PRETTY_NAME[target_metric]})"].append(
                        numpy.std(subset_results_df[target_metric])
                    )

            # Add any other columns if passed
            for column in additional_columns:
                test_scores[column].append(subset_results_df[subset_results_df["run"] == best_val_run][column].iloc[0])

    return test_scores


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
            score = Histories._compute_metric(
                metric=metric, y_pred=torch.tensor(df["adjusted_pred"].to_numpy()),
                y_true=torch.tensor(df["ground_truth"].to_numpy())
            )
            test_metrics[dataset][metric] = round(score, 3)  # This is what is used for non-refit intercepts

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
        score = Histories._compute_metric(
            metric=metric, y_pred=torch.tensor(df["adjusted_pred"].to_numpy()),
            y_true=torch.tensor(df["ground_truth"].to_numpy())
        )
        test_metrics["Pooled"][metric] = round(score, 3)  # This is what is used for non-refit intercepts
    return test_metrics


def _get_lodo_refit_scores(path, epoch, metrics):
    test_predictions = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))

    # Check the number of datasets in the test set
    datasets = set(test_predictions["dataset"])
    if len(datasets) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case for the path {path}. Found {set(test_predictions['dataset'])}")
    dataset_name = tuple(datasets)[0]

    # Average the predictions per EEG epoch
    df = get_average_prediction(test_predictions, epoch=epoch)

    # Add the targets
    target = "age"  # quick-fixed hard coding
    df["ground_truth"] = get_dataset(dataset_name).load_targets(target=target, subject_ids=test_predictions["sub_id"])

    # Estimate the intercept
    new_intercept = _estimate_intercept(df=df)
    df["adjusted_pred"] = df["pred"] + new_intercept

    # Add the performance
    test_metrics: Dict[str, float] = dict()
    for metric in metrics:
        # Normally, I'd add a 'compute_metric' method to Histories, but I don't like to change the code too much after
        # getting the results from a scientific paper, even when it makes sense. So, violating some best practice
        # instead
        score = Histories._compute_metric(
            metric=metric, y_pred=torch.tensor(df["adjusted_pred"]), y_true=torch.tensor(df["ground_truth"])
        )
        test_metrics[metric] = round(score, 3)  # This is what is used for non-refit intercepts
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
