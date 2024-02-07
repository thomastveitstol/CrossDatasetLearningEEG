"""
Script for running both leave-one-dataset-out cross validation and k-fold cross validation using the same config file
"""
import copy
import os
import random
import shutil
from datetime import datetime, date

import yaml

from cdl_eeg.data.paths import get_results_dir, get_numpy_data_storage_path
from cdl_eeg.models.random_search.generate_config_file import generate_config_file
from cdl_eeg.models.random_search.run_single_experiment import run_experiment


def main():
    # ---------------
    # Load domains of design choices/hyperparameters (.yml file)
    # ---------------
    config_path = "hyperparameter_random_search.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Create path and folder
    results_path = os.path.join(get_results_dir(),
                                f"debug_experiments_{date.today()}_{datetime.now().strftime('%H%M%S')}")
    os.mkdir(results_path)

    # ---------------
    # Generate config file
    # ---------------
    config = generate_config_file(config)  # This config file will be saved after selecting pre-processing version

    # ---------------
    # Select data pre-processing version
    # ---------------
    # Make selection. We will use the same for all datasets
    available_versions = os.listdir(get_numpy_data_storage_path())
    available_versions = tuple(version for version in available_versions if version[:5] == "debug")  # todo
    pre_processed_version = random.choice(available_versions)

    # Add selection to the datasets
    datasets = tuple(config["Training"]["Datasets"])  # Maybe this is not needed, but it feels safe
    for dataset in datasets:
        config["Training"]["Datasets"][dataset]["pre_processed_version"] = pre_processed_version

    # Add the preprocessing config file to results folder
    shutil.copy(src=os.path.join(get_numpy_data_storage_path(), pre_processed_version, "preprocessing_config.yml"),
                dst=results_path)

    # ---------------
    # Run experiments
    # ---------------
    # First, store the config file
    with open(os.path.join(results_path, "config.yml"), "w") as file:
        yaml.safe_dump(config, file)

    # Leave-one-dataset-out
    lodo_config = copy.deepcopy(config)
    lodo_config["SubjectSplit"] = {"kwargs": {"seed": 42}, "name": "SplitOnDataset"}  # todo
    if lodo_config["DomainDiscriminator"] is not None:
        num_train_datasets = len(lodo_config["Datasets"]) - 1
        lodo_config["DomainDiscriminator"]["discriminator"]["kwargs"]["num_classes"] = num_train_datasets
    run_experiment(lodo_config, results_path=os.path.join(results_path, "leave_one_dataset_out"))

    # k-fold CV
    k_fold_config = copy.deepcopy(config)
    k_fold_config["SubjectSplit"] = {"kwargs": {"seed": 42, "num_folds": len(config["Training"]["Datasets"])},
                                     "name": "KFoldDataSplit"}  # todo
    if k_fold_config["DomainDiscriminator"] is not None:
        num_train_datasets = len(k_fold_config["Datasets"])
        k_fold_config["DomainDiscriminator"]["discriminator"]["kwargs"]["num_classes"] = num_train_datasets
    run_experiment(k_fold_config, results_path=os.path.join(results_path, "k_fold_cv"))


if __name__ == "__main__":
    main()
