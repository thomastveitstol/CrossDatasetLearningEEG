"""
Script for checking performance on dataset for normal leave-one-dataset-out cross validation with different
hyperparameter values
"""
import dataclasses
import os
from enum import Enum
from typing import List, Any, Tuple, Union, Set, Dict

import numpy
import pandas
import seaborn
import yaml
from matplotlib import pyplot

from cdl_eeg.data.paths import get_results_dir

INV_FREQUENCY_BANDS = {(0.5, 4.0): "Delta",
                       (4.0, 8.0): "Theta",
                       (8.0, 12.0): "Alpha",
                       (12.0, 30.0): "Beta",
                       (30.0, 45.0): "Gamma",
                       (0.5, 45.0): "All"}


# --------------
# Convenient dataclass
# --------------
class VariableType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


@dataclasses.dataclass(frozen=True)
class HParam:
    key_path: Union[str, Tuple[str, ...]]
    preprocessing: bool  # Indicating if the hyperparameter is in the preprocessing config file or not
    default: Any
    variable_type: VariableType
    scale: str = "linear"


# --------------
# Functions for getting hyperparameter values
# --------------
def _get_config_file(results_folder, preprocessing):
    file_name = "preprocessing_config.yml" if preprocessing else "config.yml"
    with open(os.path.join(results_folder, file_name)) as f:
        config = yaml.safe_load(f)
    return config


def _get_num_montage_splits(config, hparam: HParam):
    try:
        num_montage_splits = 0
        for rbp_design in config["Varied Numbers of Channels"]["kwargs"]["RBPDesigns"].values():
            num_montage_splits += len(rbp_design["split_methods"])
    except KeyError:
        num_montage_splits = hparam.default
    return num_montage_splits


def _get_band_pass_filter(preprocessing_config):
    return INV_FREQUENCY_BANDS[tuple(preprocessing_config["general"]["filtering"])]  # type: ignore


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


def _get_weight_loss_lambda(config, hparam: HParam):
    return hparam.default if config["Training"]["Loss"]["weighter"] is None \
        else config["Training"]["Loss"]["weighter_kwargs"]["weight_power"]


def _get_hyperparameter(config, hparam: HParam):
    if hparam.key_path == "num_montage_splits":
        return _get_num_montage_splits(config=config, hparam=hparam)
    elif hparam.key_path == "band_pass":
        return _get_band_pass_filter(config)
    elif hparam.key_path == "domain_adaptation":
        return _get_domain_adaptation(config)
    elif hparam.key_path == "weighted_loss_lambda":
        return _get_weight_loss_lambda(config, hparam=hparam)
    elif hparam.key_path == "sampling_freq_multiple":
        return config["general"]["resample"] / config["general"]["filtering"][-1]
    elif hparam.key_path == "num_seconds":
        return config["general"]["num_time_steps"] / config["general"]["resample"]

    hyperparameter = config
    for key in hparam.key_path:
        try:
            hyperparameter = hyperparameter[key]

            if hyperparameter is None:
                return hparam.default
        except KeyError:
            return hparam.default

    return hyperparameter


# --------------
# Functions for getting performances
# --------------
def _get_val_test_lodo_performance(path, *, metric, balance_validation_performance):
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


def _get_val_test_inverted_lodo_performance(path, *, metric):
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
    test_df = pandas.read_csv(os.path.join(path, "sub_groups_plots", "dataset_name", metric,  # type: ignore
                                           f"test_{metric}.csv"))

    # The validation dataset (we ensured length=1 above) should not be in the test set
    if val_df.columns[0] in test_df.columns:
        raise ValueError(
            f"Expected validation dataset to not not be in the test data, but it was. Validation dataset: "
            f"{val_df.columns[0]}. Test dataset: {test_df.columns}")

    # Get performances
    test_performances: Dict[str, float] = dict()
    for dataset in test_df.columns:
        test_performances[dataset] = test_df.at[val_idx, dataset]

    # Add the balanced test performance
    test_performances["mean"] = numpy.mean(numpy.array(tuple(test_performances.values())))

    return test_performances


# --------------
# Functions for getting the correct path
# --------------
def _select_correct_lodo_fold(dataset, run_fold):
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


def _select_correct_inverted_lodo_path(source_dataset, run_fold):
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


def _get_lodo_path(run, results_dir, dataset):
    run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

    # Find out which fold is the correct one for the dataset
    fold = _select_correct_lodo_fold(dataset=dataset, run_fold=run_path)

    # Merge to obtain absolute path
    return os.path.join(run_path, fold)  # type: ignore


def _get_inverted_lodo_path(run, results_dir, source_dataset):
    run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

    # Find out which fold is the correct one for the dataset
    fold = _select_correct_inverted_lodo_path(source_dataset=source_dataset, run_fold=run_path)

    # Merge to obtain absolute path
    return os.path.join(run_path, fold)  # type: ignore


# --------------
# Hyperparameters to check
# --------------
HYPERPARAMETERS = {
    "DL Architecture": HParam(key_path=("DL Architecture", "model"), default=NotImplemented, preprocessing=False,
                              variable_type=VariableType.CATEGORICAL),
    "Spatial Dimension Handling": HParam(key_path=("Varied Numbers of Channels", "name"), default=NotImplemented,
                                         preprocessing=False, variable_type=VariableType.CATEGORICAL),
    "Learning rate": HParam(key_path=("Training", "learning_rate"), default=NotImplemented, preprocessing=False,
                            variable_type=VariableType.NUMERICAL, scale="log"),
    "Beta 1": HParam(key_path=("Training", "beta_1"), default=NotImplemented, preprocessing=False,
                     variable_type=VariableType.NUMERICAL, scale="linear"),
    "Beta 2": HParam(key_path=("Training", "beta_2"), default=NotImplemented, preprocessing=False,
                     variable_type=VariableType.NUMERICAL, scale="log"),
    "Eps": HParam(key_path=("Training", "eps"), default=NotImplemented, preprocessing=False,
                  variable_type=VariableType.NUMERICAL, scale="log"),
    "Band-pass filter": HParam(key_path="band_pass", preprocessing=True, default=NotImplemented,
                               variable_type=VariableType.CATEGORICAL),
    "Domain Adaptation": HParam(key_path="domain_adaptation", preprocessing=False,
                                variable_type=VariableType.CATEGORICAL, default=NotImplemented),
    r"Weighted loss ($\lambda$)": HParam(key_path="weighted_loss_lambda", preprocessing=False,
                                         variable_type=VariableType.NUMERICAL, default=0),
    "Number of montage splits": HParam(key_path="num_montage_splits", default=None, preprocessing=False,
                                       variable_type=VariableType.NUMERICAL),
    "Sampling freq (multiple of fmax)": HParam(key_path="sampling_freq_multiple", default=NotImplemented,
                                               preprocessing=True, variable_type=VariableType.NUMERICAL),
    "Inception depth": HParam(key_path=("DL Architecture", "kwargs", "depth"), default=None, preprocessing=False,
                              variable_type=VariableType.NUMERICAL),
    "Remove above STD": HParam(key_path=("general", "remove_above_std"), default=NotImplemented, preprocessing=True,
                               variable_type=VariableType.CATEGORICAL),
    "Time series length (s)": HParam(key_path="num_seconds", default=NotImplemented, preprocessing=True,
                                     variable_type=VariableType.NUMERICAL)
}

PRETTY_NAME = {"auc": "AUC",
               "mean": "Mean",
               "hatlestad_hall": "HatlestadHall",
               "yulin_wang": "YulinWang",
               "rockhill": "Rockhill",
               "mpi_lemon": "MPI Lemon",
               "miltiadous": "Miltiadous"}

# Cosmetics
FONTSIZE = 17
FIGSIZE = (16, 9)


# --------------
# Main functions
# --------------
def main_lodo():
    folder_name = "easter_runs"  # "debug_results_1"

    # --------------
    # Hyperparameters
    # --------------
    hyperparameter_name = "Inception depth"
    hyperparam = HYPERPARAMETERS[hyperparameter_name]

    dataset = "mpi_lemon"
    performance_metric = "auc"
    balance_validation_performance = True

    results_dir = os.path.join(get_results_dir(), folder_name)

    # --------------
    # Select runs
    # --------------
    runs = (run for run in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, run,
                                                                                  "leave_one_dataset_out",
                                                                                  "finished_successfully.txt"))
            and "inverted_cv" not in run)

    # --------------
    # Get performances
    # --------------
    val_performance: List[float] = []
    test_performance: List[float] = []
    hyperparameter: List[Any] = []
    for run in runs:
        # Get the path
        path = _get_lodo_path(run=run, results_dir=results_dir, dataset=dataset)

        # Get performance
        val, test = _get_val_test_lodo_performance(path=path, metric=performance_metric,
                                                   balance_validation_performance=balance_validation_performance)
        val_performance.append(val)
        test_performance.append(test)
        hyperparameter.append(_get_hyperparameter(
            config=_get_config_file(results_folder=os.path.dirname(os.path.dirname(path)),
                                    preprocessing=hyperparam.preprocessing),
            hparam=hyperparam
        ))

    # --------------
    # Plotting
    # --------------
    pyplot.figure(figsize=FIGSIZE)

    if hyperparam.variable_type == VariableType.NUMERICAL:
        pyplot.scatter(x=hyperparameter, y=test_performance)

        pyplot.xlabel(hyperparameter_name, fontsize=FONTSIZE)
        pyplot.ylabel(f"Test ({PRETTY_NAME[performance_metric]})", fontsize=FONTSIZE)

    elif hyperparam.variable_type == VariableType.CATEGORICAL:
        data = pandas.DataFrame.from_dict({hyperparameter_name: hyperparameter, "Performance": test_performance})
        ax = seaborn.boxplot(data, y=hyperparameter_name, x="Performance", hue=hyperparameter_name)

        ax.xaxis.label.set_size(FONTSIZE)
        ax.yaxis.label.set_size(FONTSIZE)
    else:
        raise ValueError(f"Unexpected variable type: {hyperparam.variable_type}")

    # Cosmetics
    pyplot.xscale(hyperparam.scale)

    pyplot.xticks(fontsize=FONTSIZE)
    pyplot.yticks(fontsize=FONTSIZE)
    pyplot.title(f"{PRETTY_NAME[dataset]}", fontsize=FONTSIZE+3)

    pyplot.show()


def main_inverted_lodo():
    folder_name = "easter_runs"

    # --------------
    # Hyperparameters
    # --------------
    hyperparameter_name = "Time series length (s)"
    hyperparam = HYPERPARAMETERS[hyperparameter_name]

    # Datasets
    source_dataset = "rockhill"
    performance_metric = "auc"
    jitter = None

    alpha = 0.2

    results_dir = os.path.join(get_results_dir(), folder_name)

    # --------------
    # Select runs
    # --------------
    runs = (run for run in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, run,
                                                                                  "leave_one_dataset_out",
                                                                                  "finished_successfully.txt"))
            and "inverted_cv" in run)

    # --------------
    # Get performances
    # --------------
    test_performances: Dict[str, List[float]] = dict()
    hyperparameter: List[Any] = []
    for i, run in enumerate(runs):
        # Get the path
        path = _get_inverted_lodo_path(run=run, results_dir=results_dir, source_dataset=source_dataset)

        # Get performance
        test = _get_val_test_inverted_lodo_performance(path=path, metric=performance_metric)

        # Store performance
        for dataset_name, performance in test.items():
            if i == 0:
                test_performances[dataset_name] = [performance]
            else:
                test_performances[dataset_name].append(performance)

        hyperparameter.append(_get_hyperparameter(
            config=_get_config_file(results_folder=os.path.dirname(os.path.dirname(path)),
                                    preprocessing=hyperparam.preprocessing),
            hparam=hyperparam
        ))

    # --------------
    # Plotting
    # --------------
    pyplot.figure(figsize=FIGSIZE)

    if hyperparam.variable_type == VariableType.NUMERICAL:
        for dataset_name, test_performance in test_performances.items():
            x = hyperparameter.copy()
            if isinstance(jitter, (int, float)):
                x = [val + numpy.random.normal(0, jitter) if val is not None else None for val in x]
            pyplot.scatter(x=x, y=test_performance, label=PRETTY_NAME[dataset_name],
                           alpha=1 if dataset_name == "mean" else alpha)

        pyplot.xlabel(hyperparameter_name, fontsize=FONTSIZE)
        pyplot.ylabel(f"Test ({PRETTY_NAME[performance_metric]})", fontsize=FONTSIZE)
        pyplot.legend(fontsize=FONTSIZE)
    elif hyperparam.variable_type == VariableType.CATEGORICAL:
        data: Dict[str, List[Any]] = {hyperparameter_name: [], "Performance": [], "Target dataset": []}
        for dataset_name, test_performance in test_performances.items():
            for performance, hyperparameter_value in zip(test_performance, hyperparameter):
                data[hyperparameter_name].append(hyperparameter_value)
                data["Performance"].append(performance)
                data["Target dataset"].append(PRETTY_NAME[dataset_name])

        ax = seaborn.boxplot(data, x="Performance", y="Target dataset", hue=hyperparameter_name)
        # ax = seaborn.boxplot(data, x="Performance", y=hyperparameter_name, hue="Target dataset")

        ax.xaxis.label.set_size(FONTSIZE)
        ax.yaxis.label.set_size(FONTSIZE)
        ax.legend(fontsize=FONTSIZE)
    else:
        raise ValueError(f"Unexpected variable type: {hyperparam.variable_type}")

    # Cosmetics
    pyplot.xscale(hyperparam.scale)

    pyplot.xticks(fontsize=FONTSIZE)
    pyplot.yticks(fontsize=FONTSIZE)
    pyplot.title(f"Source dataset: {PRETTY_NAME[source_dataset]}", fontsize=FONTSIZE+3)

    pyplot.show()


def main():
    main_inverted_lodo()


if __name__ == "__main__":
    main()
