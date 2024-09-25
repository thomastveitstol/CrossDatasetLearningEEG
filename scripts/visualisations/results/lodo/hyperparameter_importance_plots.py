"""
Functions and stuff for analysing the impact of different hyperparameters using Optuna

I think the best way is to define a 'Study' object
"""
import dataclasses
import os
import random
from typing import Dict, Tuple, Union, Optional, Set

import numpy
import optuna
import pandas
import seaborn
import yaml
from matplotlib import pyplot, rcParams
from optuna import Study
from optuna.distributions import BaseDistribution, FloatDistribution, CategoricalDistribution, IntDistribution
from optuna.importance import get_param_importances, FanovaImportanceEvaluator

from cdl_eeg.data.analysis.results_analysis import get_config_file, get_lodo_dataset_name, SkipFold, higher_is_better, \
    get_all_lodo_runs, PRETTY_NAME
from cdl_eeg.data.paths import get_results_dir


@dataclasses.dataclass(frozen=True)
class _HP:
    key_path: Union[Tuple[str, ...], str]
    preprocessing: bool
    dist_path: Optional[Union[Tuple[str, ...], str]] = None
    in_dist_config: bool = True  # If the distribution is to be found in the config file. If False, special case must be
    # implemented
    is_conditional: bool = False  # If this is True, we will interpret KeyError raised when the HP is not found. But I
    # don't think I can use it... :(



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


def _get_spatial_method(config):
    method = config["Varied Numbers of Channels"]["name"]
    if method == "RegionBasedPooling":
        return "RBP"
    elif method == "Interpolation":
        return config["Varied Numbers of Channels"]["kwargs"]["method"]
    else:
        raise ValueError(f"Unexpected method for handling varied numbers of channels: {method}")


def _get_band_pass_filter(preprocessing_config):
    return INV_FREQUENCY_BANDS[tuple(preprocessing_config["general"]["filtering"])]  # type: ignore


def _get_normalisation(config):
    if config["Varied Numbers of Channels"]["name"] == "Interpolation":
        return config["DL Architecture"]["normalise"]
    elif config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling":
        return config["Varied Numbers of Channels"]["kwargs"]["normalise_region_representations"]
    else:
        raise ValueError


def _get_weighted_loss(config):
    loss_config = config["Training"]["Loss"]  # tau in the paper (as per now, at least)

    # Not using weighted loss is equivalent to tau = 0
    return 0 if loss_config["weighter"] is None else loss_config["weighter_kwargs"]["weight_power"]


def _get_hyperparameter(config, hparam: _HP, dist):
    if hparam.key_path == "domain_adaptation" and not dist:
        return _get_domain_adaptation(config)
    elif hparam.key_path == "spatial_dimension" and not dist:
        return _get_spatial_method(config)
    elif hparam.key_path == "band_pass":
        return _get_band_pass_filter(config)
    elif hparam.key_path == "sampling_freq_multiple":
        return f"{int(config['general']['resample'] // config['general']['filtering'][-1])} * f max"
    elif hparam.key_path == "normalisation":
        return _get_normalisation(config)
    elif hparam.key_path == "num_seconds":
        return f"{int(config['general']['num_time_steps'] // config['general']['resample'])}s"
    elif hparam.key_path == "weighted_loss":
        return _get_weighted_loss(config)

    hyperparameter = config.copy()
    keys = hparam.dist_path if dist and hparam.dist_path is not None else hparam.key_path
    for key in keys:
        hyperparameter = hyperparameter[key]
    return hyperparameter


def _get_params(path, hyperparameters):
    """Generates parameters to be passed to the Optuna create_trial"""
    config = get_config_file(results_folder=path, preprocessing=False)  # todo
    preprocessing_config = get_config_file(results_folder=path, preprocessing=True)

    hparams = dict()
    for hp_name, hp_value in hyperparameters.items():
        try:
            _config = preprocessing_config if hp_value.preprocessing else config
            hparams[hp_name] = _get_hyperparameter(config=_config, hparam=hp_value, dist=False)
        except KeyError as e:
            if not hp_value.is_conditional:
                raise KeyError(f"Could not find the HP {hp_name}. If this is expected (conditional HP), "
                               f"set is_conditional to True") from e

    return hparams


# ----------------
# Functions for getting the results
# todo: hard-coded LODO
# ----------------
def _get_validation_performance(path, *, main_metric, balance_validation_performance):
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

    return val_metric, val_idx


def _get_test_performance(path,  *, target_metric, selection_metric, datasets, balance_validation_performance):
    """Function for getting the test score"""
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
    val_performance, epoch = _get_validation_performance(
        path=path, main_metric=selection_metric, balance_validation_performance=balance_validation_performance
    )

    # --------------
    # Get test performance
    # --------------
    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_performance = test_df[target_metric][epoch]

    return test_performance, dataset_name


def _create_trial(parameters, distributions, score):
    """Create a trial to be added to the study"""
    score = 0 if numpy.isnan(score) else score  # todo: not a good solution, what if 0 is good?

    # ----------------
    # Remove unused HPs from distribution to avoid ValueError for conditional HPs
    # ----------------
    reduced_distributions = {param_name: distributions[param_name] for param_name in parameters}

    # ----------------
    # Create trial object
    # ----------------
    trial = optuna.trial.create_trial(
        params=parameters,
        distributions=reduced_distributions,
        value=score
    )
    return trial


def _get_mts_module_dist(distribution):
    return CategoricalDistribution(choices=distribution.keys())  # Not very elegant, but it works


def _get_domain_adaptation_dist():
    return CategoricalDistribution(choices=("CMMN + DD", "CMMN", "DD", "Nothing"))


def _get_spatial_dim_choices():
    # I don't think I need to specify the weights
    return CategoricalDistribution(choices=("spline", "MNE", "RBP"))


def _get_band_pass_dist():
    return CategoricalDistribution(choices=list(INV_FREQUENCY_BANDS.values()))


def _get_sampling_freq_dist():
    return CategoricalDistribution(choices=("2 * f max", "4 * f max", "8 * f max"))


def _get_normalisation_dist():
    return CategoricalDistribution(choices=(True, False))


def _get_input_length_dist():
    return CategoricalDistribution(choices=("5s", "10s"))


def _get_loss_dist():
    return CategoricalDistribution(choices=("L1Loss", "MSELoss"))


def _get_weighted_loss_dist():
    # Although it is not uniform distribution, it is still a float distribution
    return FloatDistribution(low=0, high=1)


def _get_autoreject_dist():
    return CategoricalDistribution(choices=(True, False))


def _config_dist_to_optuna_dist(distribution, hp_name):
    """Convert from how the distributions are specified in the sampling config file to Optuna compatible instances"""
    if hp_name == "DL Architecture":
        return _get_mts_module_dist(distribution)
    elif hp_name == "DomainAdaptation":
        return _get_domain_adaptation_dist()
    elif hp_name == "Spatial Dimension Handling":
        return _get_spatial_dim_choices()
    elif hp_name == "Band pass":
        return _get_band_pass_dist()
    elif hp_name == "Sampling frequency":
        return _get_sampling_freq_dist()
    elif hp_name == "Normalisation":
        return _get_normalisation_dist()
    elif hp_name == "Input length":
        return _get_input_length_dist()
    elif hp_name == "Loss":
        return _get_loss_dist()
    elif hp_name == r"Weighted loss ($\tau$)":
        return _get_weighted_loss_dist()
    elif hp_name == "Autoreject":
        return _get_autoreject_dist()

    # Not a very elegant function... should preferably use the sampling distribution functions
    dist_name = distribution["dist"]
    if dist_name == "uniform":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return FloatDistribution(low=low, high=high)
    elif dist_name == "log_uniform":
        base = distribution["kwargs"]["base"]
        low = base ** distribution["kwargs"]["a"]
        high = base ** distribution["kwargs"]["b"]
        return FloatDistribution(low=low, high=high, log=True)
    if dist_name == "uniform_int":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return IntDistribution(low=low, high=high)
    elif dist_name == "log_uniform_int":
        base = distribution["kwargs"]["base"]
        low = base ** distribution["kwargs"]["a"]
        high = base ** distribution["kwargs"]["b"]
        return IntDistribution(low=low, high=high, log=True)
    elif dist_name == "n_log_uniform_int":
        # I'll register it as a log scale
        n = distribution["kwargs"]["n"]
        base = distribution["kwargs"]["base"]
        low = n * round(base ** distribution["kwargs"]["a"])
        high = n * round(base ** distribution["kwargs"]["b"])
        return IntDistribution(low=low, high=high, log=True)
    else:
        raise ValueError(f"Unrecognised distribution: {dist_name}")


def _get_optuna_distributions(hyperparameters, dist_config) -> Dict[str, BaseDistribution]:
    """Function for getting the distributions spaces of the HPs, compatible with Optuna"""
    # --------------
    # Loop through the desired hyperparameters
    # --------------
    optuna_hp_distributions: Dict[str, BaseDistribution] = dict()
    for hp_name, hp_info in hyperparameters.items():
        if hp_info.in_dist_config:
            hp_dist = _get_hyperparameter(dist_config, hparam=hp_info, dist=True)
        else:
            hp_dist = None
        optuna_hp_distributions[hp_name] = _config_dist_to_optuna_dist(distribution=hp_dist, hp_name=hp_name)

    return optuna_hp_distributions


def create_studies(*, datasets, runs, direction, results_dir, target_metric, selection_metric,
                   balance_validation_performance, dist_config, hyperparameters=None):
    """
    Function for creating 'Study' objects per dataset using Optuna. Once these are obtained, hyperparameter importance
    can be assessed

    Parameters
    ----------
    datasets : tuple[str, ...]
        Dataset names we want to create studies of
    runs : tuple[str, ...]
        Runs we want to add to the studies
    direction
        Indicates if the objective should be maximised or minimised
    results_dir
        Path to where the results are stored
    hyperparameters
        The hyperparameters to register as part of the trials
    target_metric
        Metric to optimise
    selection_metric
        Metric for model selection (epoch selection in this case)
    balance_validation_performance : bool
    dist_config
        configurations file containing distributions used

    Returns
    -------
    dict[str, Study]
    """
    # -------------
    # Extract the original HP distributions
    #
    # This is because Optuna relies on knowing the distributions
    # we sampled from, not only the HP configurations.
    # -------------
    hyperparameters = _HYPERPARAMETERS.copy() if hyperparameters is None else hyperparameters
    distributions: Dict[str, BaseDistribution] = _get_optuna_distributions(
        hyperparameters=hyperparameters, dist_config=dist_config
    )

    # -------------
    # Create Optuna studies (one per dataset)
    # -------------
    studies: Dict[str, Study] = {dataset_name: optuna.create_study(direction=direction) for dataset_name in datasets}
    for run in runs:
        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the hyperparameters of the current run/experiment
        parameters = _get_params(path=os.path.dirname(run_path), hyperparameters=hyperparameters)

        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            # Get the performance
            try:
                test_performance, dataset_name = _get_test_performance(
                    path=os.path.join(run_path, fold), target_metric=target_metric,  # type: ignore
                    selection_metric=selection_metric, datasets=datasets,
                    balance_validation_performance=balance_validation_performance
                )
            except SkipFold:
                continue
            except KeyError:
                # todo: how to register seriously bad runs? This error should only occur for correlations, so could set
                #  it to zero?
                continue

            # Compute the trial
            trial = _create_trial(parameters, distributions, score=test_performance)

            # Add the trial to the study object of the current dataset
            studies[dataset_name].add_trial(trial)

    return studies


INV_FREQUENCY_BANDS = {(1.0, 4.0): "Delta",
                       (4.0, 8.0): "Theta",
                       (8.0, 12.0): "Alpha",
                       (12.0, 30.0): "Beta",
                       (30.0, 45.0): "Gamma",
                       (1.0, 45.0): "All"}


_HYPERPARAMETERS = {
    # Training HPs
    "Learning rate": _HP(key_path=("Training", "learning_rate"), preprocessing=False),
    r"$\beta_1$": _HP(key_path=("Training", "beta_1"), preprocessing=False),
    r"$\beta_2$": _HP(key_path=("Training", "beta_2"), preprocessing=False),
    r"$\epsilon$": _HP(key_path=("Training", "eps"), preprocessing=False),
    # DL architecture
    "DL Architecture": _HP(key_path=("DL Architecture", "model"), dist_path=("MTS Module",), preprocessing=False),
    #"IN:depth": _HP(key_path=("DL Architecture", "kwargs", "depth"), dist_path=("InceptionNetwork", "sample", "depth"),
    #                preprocessing=False, is_conditional=True),
    #"IN:cnn_units": _HP(key_path=("DL Architecture", "kwargs", "cnn_units"),
    #                    dist_path=("InceptionNetwork", "sample", "cnn_units"), preprocessing=False,
    #                    is_conditional=True),
    #"SN:n_filters": _HP(key_path=("DL Architecture", "kwargs", "n_filters"),
    #                    dist_path=("ShallowFBCSPNetMTS", "sample", "n_filters"),
    #                    preprocessing=False, is_conditional=True),
    # Spatial dimension mismatch
    "Spatial Dimension Handling": _HP(key_path="spatial_dimension", preprocessing=False, in_dist_config=False),
    # Domain discriminator
    "DomainAdaptation": _HP(key_path="domain_adaptation", preprocessing=False, in_dist_config=False),
    # Pre-processing / feature extraction
    "Band pass": _HP(key_path="band_pass", preprocessing=True, in_dist_config=False),
    "Sampling frequency": _HP(key_path="sampling_freq_multiple", preprocessing=True, in_dist_config=False),
    "Normalisation": _HP(key_path="normalisation", preprocessing=False, in_dist_config=False),
    "Input length": _HP(key_path="num_seconds", preprocessing=True, in_dist_config=False),
    "Autoreject": _HP(key_path=("general", "autoreject"), preprocessing=True, in_dist_config=False),
    # Loss
    "Loss": _HP(key_path=("Training", "Loss", "loss"), preprocessing=False, in_dist_config=False),
    r"Weighted loss ($\tau$)": _HP(key_path="weighted_loss", preprocessing=False, in_dist_config=False)
}


EVALUATORS = {"fanova": FanovaImportanceEvaluator()}
FONTSIZE = 18
FIGSIZE = (20, 17)
Y_ROTATION = None
ERRORBAR = "sd"
rcParams["legend.fontsize"] = FONTSIZE
rcParams["legend.title_fontsize"] = FONTSIZE
rcParams["axes.labelsize"] = FONTSIZE + 2


def main():
    meaning_of_life = 42

    random.seed(meaning_of_life)
    numpy.random.seed(meaning_of_life)

    # ----------------
    # A few design choices for the analysis
    # ----------------
    num_evaluations = 5  # Running fANOVA does not always yield the same results. This determines the number of times to
    # run HP importance
    datasets = ("TDBrain", "MPILemon", "HatlestadHall")
    evaluator = "fanova"
    results_dir = get_results_dir()
    selection_metric = "mae"
    target_metric = "pearson_r"
    direction = "maximize" if higher_is_better(target_metric) else "minimize"
    balance_validation_performance = False
    hyperparameters = _HYPERPARAMETERS
    runs = get_all_lodo_runs(results_dir=results_dir, successful_only=True)  # [:40]
    config_dist_path = os.path.join(  # todo: not very elegant...
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "training",
        "config_files", "hyperparameter_random_search.yml"
    )
    with open(config_dist_path) as f:
        config_dist = yaml.safe_load(f)

    # ----------------
    # Create the studies
    # ----------------
    studies = create_studies(
        datasets=datasets, runs=runs, direction=direction, results_dir=results_dir, target_metric=target_metric,
        selection_metric=selection_metric, balance_validation_performance=balance_validation_performance,
        hyperparameters=hyperparameters, dist_config=config_dist
    )

    # ----------------
    # Get the hyperparameter importance scores
    # ----------------
    print(f"Evaluator: {evaluator}")
    importances = {"Dataset": [], "Importance": [], "Hyperparameter": []}
    for i in range(num_evaluations):
        print(f"\nEvaluation {i + 1}/{num_evaluations}")
        for dataset_name, study in studies.items():
            # Maybe some printing
            if i == 0:
                print(f"\n=== {PRETTY_NAME[dataset_name]} ===")
                for param_name, param_value in study.best_params.items():
                    print(f"{param_name}: {param_value}")

            # Get HP importance
            hp_importance = get_param_importances(study, evaluator=EVALUATORS[evaluator])

            # Add to dict
            for param_name, importance in hp_importance.items():
                importances["Dataset"].append(PRETTY_NAME[dataset_name])
                importances["Importance"].append(importance)
                importances["Hyperparameter"].append(param_name)

    # ----------------
    # Plotting
    # ----------------
    df = pandas.DataFrame.from_dict(importances)
    pyplot.figure(figsize=FIGSIZE, layout="constrained")
    seaborn.barplot(df, hue="Dataset", y="Hyperparameter", x="Importance", order=hyperparameters.keys(),
                    errorbar=ERRORBAR)

    # Cosmetics
    pyplot.grid(axis="x")
    pyplot.xlim(0, None)
    pyplot.tick_params(labelsize=FONTSIZE)
    pyplot.yticks(rotation=Y_ROTATION)

    pyplot.show()


if __name__ == "__main__":
    main()
