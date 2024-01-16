"""
Script for running both leave-one-dataset-out cross validation and k-fold cross validation using the same config file
"""
import copy
import os
from datetime import datetime

import yaml

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.models.experiments.generate_config_file import generate_config_file
from cdl_eeg.models.experiments.run_single_experiment import run_experiment


def main():
    # ---------------
    # # Load domains of design choices/hyperparameters (.yml file)
    # ---------------
    config_path = "hyperparameter_random_search.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Create path and folder
    results_path = os.path.join(get_results_dir(), f"debug_experiments_{datetime.now().strftime('%H%M%S')}")
    os.mkdir(results_path)

    # ---------------
    # Generate config file
    # ---------------
    config = generate_config_file(config)

    with open(os.path.join(results_path, "config.yml"), "w") as file:
        yaml.safe_dump(config, file)

    # ---------------
    # Run experiments
    # ---------------
    # Leave-one-dataset-out
    lodo_config = copy.deepcopy(config)
    lodo_config["Training"]["Data Split"] = {"kwargs": {"seed": 42}, "name": "SplitOnDataset"}
    if lodo_config["DomainDiscriminator"] is not None:
        num_train_datasets = len(lodo_config["Training"]["Datasets"]) - 1
        lodo_config["DomainDiscriminator"]["discriminator"]["kwargs"]["num_classes"] = num_train_datasets
    run_experiment(lodo_config, results_path=os.path.join(results_path, "leave_one_dataset_out"))

    # k-fold CV
    k_fold_config = copy.deepcopy(config)
    k_fold_config["Training"]["Data Split"] = {"kwargs": {"seed": 42, "num_folds": len(config["Training"]["Datasets"])},
                                               "name": "KFoldDataSplit"}
    if k_fold_config["DomainDiscriminator"] is not None:
        k_fold_config["DomainDiscriminator"]["kwargs"]["num_classes"] = len(k_fold_config["Training"]["Datasets"])
    run_experiment(k_fold_config, results_path=os.path.join(results_path, "k_fold_cv"))


if __name__ == "__main__":
    main()
