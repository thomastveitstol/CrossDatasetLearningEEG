"""
Script for running both leave-one-dataset-out cross validation and k-fold cross validation using the same config file
"""
import copy
import os
import random
from datetime import datetime, date

import yaml

from cdl_eeg.data.paths import get_results_dir, get_numpy_data_storage_path
from cdl_eeg.models.random_search.generate_config_file import generate_config_file
from cdl_eeg.models.random_search.run_single_experiment import Experiment


def _str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError(f"Unexpected string: {s}")


def main():
    # ---------------
    # Load domains of design choices/hyperparameters (.yml file)
    # ---------------
    config_path = "hyperparameter_random_search.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Create path and folder
    cv_method = random.choice(config["cv_method"])
    config["cv_method"] = cv_method
    if cv_method == "normal":
        _title_cv = "cv"
    elif cv_method == "inverted":
        _title_cv = "inverted_cv"
    else:
        raise ValueError
    results_path = os.path.join(get_results_dir(),
                                f"debug_{config['selected_target']}_{_title_cv}_experiments_{date.today()}_"
                                f"{datetime.now().strftime('%H%M%S')}")
    os.mkdir(results_path)

    # ---------------
    # Generate config file
    # ---------------
    config = generate_config_file(config)  # This config file will be saved after selecting pre-processing version

    # ---------------
    # Select data pre-processing version
    # ---------------
    # Make selection. We will use the same for all datasets
    # todo: Not sure how I feel about this hard-coding
    preprocessed_folder = "preprocessed_2024-04-10_103043"
    available_versions = os.listdir(os.path.join(get_numpy_data_storage_path(), preprocessed_folder))
    available_versions = tuple(version for version in available_versions if version[:5] == "data_")
    pre_processed_folder = random.choice(available_versions)
    pre_processed_version = os.path.join(preprocessed_folder, pre_processed_folder)

    # Add selection to the datasets
    datasets = tuple(config["Datasets"])  # Maybe this is not needed, but it feels safe
    for dataset in datasets:
        config["Datasets"][dataset]["pre_processed_version"] = pre_processed_version

    # Extract preprocessing config
    with open(os.path.join(get_numpy_data_storage_path(), preprocessed_folder, "config.yml"), "r") as file:
        pre_processed_config = yaml.safe_load(file)

    # Add details
    filtering = pre_processed_folder.split("_")[3].split(sep="-")
    l_freq, h_freq = float(filtering[0]), float(filtering[1])
    pre_processed_config["general"]["filtering"] = [l_freq, h_freq]
    pre_processed_config["general"]["autoreject"] = _str_to_bool(pre_processed_folder.split("_")[5])
    s_freq = float(pre_processed_folder.split("_")[-1]) * h_freq
    pre_processed_config["general"]["resample"] = float(pre_processed_folder.split("_")[-1]) * h_freq
    pre_processed_config["general"]["num_time_steps"] = int(s_freq * pre_processed_config["general"]["epoch_duration"])
    del pre_processed_config["frequency_bands"]
    del pre_processed_config["resample_fmax_multiples"]

    # Save it
    with open(os.path.join(results_path, "preprocessing_config.yml"), "w") as file:
        yaml.safe_dump(pre_processed_config, file)

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

    print(f"\n{' Leave-one-dataset-out cross validation ':=^50}\n")
    leave_one_dataset_out_experiment = Experiment(config=lodo_config, pre_processing_config=pre_processed_config,
                                                  results_path=os.path.join(results_path, "leave_one_dataset_out"))
    leave_one_dataset_out_experiment.run_experiment()

    if config["run_baseline"]:
        # k-fold CV
        k_fold_config = copy.deepcopy(config)
        k_fold_config["SubjectSplit"] = {"kwargs": {"seed": 42, "num_folds": len(config["Datasets"])},
                                         "name": "KFoldDataSplit"}  # todo
        if k_fold_config["DomainDiscriminator"] is not None:
            num_train_datasets = len(k_fold_config["Datasets"])
            k_fold_config["DomainDiscriminator"]["discriminator"]["kwargs"]["num_classes"] = num_train_datasets

        print(f"\n{' Baseline experiment ':=^50}\n")
        k_fold_experiment = Experiment(config=k_fold_config, pre_processing_config=pre_processed_config,
                                       results_path=os.path.join(results_path, "k_fold_cv"))
        k_fold_experiment.run_experiment()


if __name__ == "__main__":
    main()
