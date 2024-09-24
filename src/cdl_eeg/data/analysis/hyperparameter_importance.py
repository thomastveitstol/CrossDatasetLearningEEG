"""
Functions and stuff for analysing the impact of different hyperparameters using Optuna

I think the best way is to define a 'Study' object
"""
import dataclasses
import os
from typing import Dict, Tuple, Union

import numpy
import optuna
import pandas
from optuna import Study
from optuna.distributions import BaseDistribution, FloatDistribution

from cdl_eeg.data.analysis.results_analysis import get_config_file, get_lodo_dataset_name, SkipFold, higher_is_better


@dataclasses.dataclass(frozen=True)
class _HP:
    key_path: Union[Tuple[str, ...], str]
    preprocessing: bool


def _get_hyperparameter(config, hparam: _HP):
    hyperparameter = config.copy()
    for key in hparam.key_path:
        hyperparameter = hyperparameter[key]
    return hyperparameter


def _get_params(path, hyperparameters):
    """Generates parameters to be passed to the Optuna create_trial"""
    config = get_config_file(results_folder=path, preprocessing=False)  # todo

    return {hp_name: _get_hyperparameter(config=config, hparam=hp_value)
            for hp_name, hp_value in hyperparameters.items()}


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
    trial = optuna.trial.create_trial(
        params=parameters,
        distributions=distributions,
        value=score
    )
    return trial


_HYPERPARAMETERS = {
    # Training HPs
    # "learning_rate": _HP(key_path=("Training", "learning_rate"), preprocessing=False),
    "beta_1": _HP(key_path=("Training", "beta_1"), preprocessing=False),
    "beta_2": _HP(key_path=("Training", "beta_2"), preprocessing=False),
    # "eps": _HP(key_path=("Training", "eps"), preprocessing=False),
}


def _config_dist_to_optuna_dist(distribution):
    """Convert from how the distributions are specified in the sampling config file to Optuna compatible instances"""
    # Not a very elegant function...
    dist_name = distribution["dist"]
    if dist_name == "uniform":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return FloatDistribution(low=low, high=high)
    elif dist_name == "log_uniform":
        low = distribution["kwargs"]["a"]
        high = distribution["kwargs"]["b"]
        return FloatDistribution(low=low, high=high, log=True)
    else:
        raise ValueError(f"Unrecognised distribution: {dist_name}")


def _get_optuna_distributions(hyperparameters, dist_config) -> Dict[str, BaseDistribution]:
    """Function for getting the distributions spaces of the HPs, compatible with Optuna"""
    # --------------
    # Loop through the desired hyperparameters
    # --------------
    optuna_hp_distributions: Dict[str, BaseDistribution] = dict()
    for hp_name, hp_info in hyperparameters.items():
        hp_dist = _get_hyperparameter(dist_config, hparam=hp_info)
        optuna_hp_distributions[hp_name] = _config_dist_to_optuna_dist(distribution=hp_dist)

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
