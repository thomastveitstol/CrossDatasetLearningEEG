"""
Functions and stuff for analysing the impact of different hyperparameters using Optuna

I think the best way is to define a 'Study' object
"""
import os
from typing import Dict

import optuna
from optuna import Study
from optuna.distributions import BaseDistribution


def _get_params():
    """Generates parameters to be passed to the Optuna create_trial"""
    raise NotImplementedError


def _get_test_performance():
    """Function for getting the test score"""
    raise NotImplementedError


def _create_trial(parameters, distributions, score):
    """Create a trial to be added to the study"""
    trial = optuna.trial.create_trial(
        params=parameters,
        distributions=distributions,
        value=score
    )
    return trial

def _get_optuna_distributions():
    """Function for getting the distributions spaces of the HPs, compatible with Optuna"""
    raise NotImplementedError


def create_studies(datasets, runs, direction, results_dir):
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
    distributions: Dict[str, BaseDistribution] = _get_optuna_distributions()

    # -------------
    # Create Optuna studies (one per dataset)
    # -------------
    studies: Dict[str, Study] = {dataset_name: optuna.create_study(direction=direction) for dataset_name in datasets}
    for run in runs:
        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the hyperparameters of the current run/experiment
        parameters = _get_params()

        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            # Get the performance
            test_performance, dataset_name = _get_test_performance()

            # Compute the trial
            trial = _create_trial(parameters, distributions, score=test_performance)

            # Add the trial to the study object of the current dataset
            studies[dataset_name].add_trial(trial)
