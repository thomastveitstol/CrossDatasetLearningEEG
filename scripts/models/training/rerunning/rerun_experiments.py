"""
Script for re-running experiments. Added after an error in a pooling method of RBP was spotted

Implementation-wise, this requires (1) to select from the affected_runs.csv file, and (2) not selecting a run that has
already been re-run.
"""
import copy
import os
import time
from datetime import datetime, date
from pathlib import Path

import pandas
import yaml

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.models.random_search.run_single_experiment import Experiment


# --------------
# Function for suggesting the next run
# --------------
def _get_completed(*, results_dir):
    """Function for removing runs which has already been completed"""
    completed_reruns = []
    for run in os.listdir(results_dir):
        if run.endswith("_rerun"):
            completed_reruns.append(run.split("--")[0])
    return tuple(completed_reruns)


def _suggest_run(results_dir) -> str:
    # Get the list of affected experiments
    path = Path(os.path.dirname(__file__)) / "affected_runs.csv"
    affected_runs = tuple(pandas.read_csv(path)["affected"])

    # Get the list of experiments which has already been re-run
    completed_runs = _get_completed(results_dir=results_dir)

    # Get a set of new potential experiments to run
    potential_new_runs = set(affected_runs) - set(completed_runs)

    # If there are no more new runs, just sleep for a while
    if not potential_new_runs:
        time.sleep(60)
        raise RuntimeError("There are no more experiments to re-run :)")

    return next(iter(potential_new_runs))


# --------------
# Main function for re-running
# --------------
def main():
    results_root_dir = get_results_dir()

    # Get the experiment to re-run
    run = _suggest_run(results_root_dir)

    # Fix the new directory and stuff
    today = f"{date.today()}_{datetime.now().strftime('%H%M%S')}"
    experiment_path = os.path.join(results_root_dir, f"{run}--{today}_rerun")

    # Get the config files
    _previous_path = Path(results_root_dir) / run
    with open(_previous_path / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    with open(_previous_path / "preprocessing_config.yml", "r") as file:
        pre_processed_config = yaml.safe_load(file)

    # Set continuous testing to False and override the 'main_metric'
    config["Training"]["continuous_testing"] = False
    config["Training"]["main_metric"] = ("mae", "mse", "r2_score", "pearson_r")

    # Save the config files as it was originally done because it makes life easier when analysing HPs
    with open(os.path.join(experiment_path, "config.yml"), "w") as file:
        yaml.safe_dump(config, file)

    with open(os.path.join(experiment_path, "preprocessing_config.yml"), "w") as file:
        yaml.safe_dump(pre_processed_config, file)

    # Add some details for domain discriminator (as in single_experiments)
    experiment_config = copy.deepcopy(config)
    if experiment_config["DomainDiscriminator"] is not None:
        num_train_datasets = len(experiment_config["Datasets"]) - 1
        experiment_config["DomainDiscriminator"]["discriminator"]["kwargs"]["num_classes"] = num_train_datasets

    with Experiment(config=experiment_config, pre_processing_config=pre_processed_config,
                    results_path=os.path.join(experiment_path, "leave_one_dataset_out")) as experiment:
        experiment.run_experiment()


if __name__ == "__main__":
    main()
