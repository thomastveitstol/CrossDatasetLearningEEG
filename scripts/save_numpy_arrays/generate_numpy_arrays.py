"""
Script for generating numpy arrays, which pre-processing hyperparameters are sampled from a config file
"""
import os
import shutil
from datetime import date, datetime

import yaml

from cdl_eeg.data.config_file_generator import generate_preprocessing_config_file
from cdl_eeg.data.datasets.dataset_base import DataError
from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_numpy_data_storage_path


def main():
    verbose = True

    # --------------
    # Load domains of design choices/hyperparameters (.yml file)
    # --------------
    with open(os.path.join(os.path.dirname(__file__), "config_domains.yml")) as f:
        domains_config = yaml.safe_load(f)

    # Create path and folder
    path = os.path.join(get_numpy_data_storage_path(),
                        f"debug_preprocessed_{date.today()}_{datetime.now().strftime('%H%M%S')}")
    os.mkdir(path)

    # --------------
    # Generate config file
    # --------------
    # Generate config file
    config = generate_preprocessing_config_file(domains_config)

    # Save config file
    with open(os.path.join(path, "preprocessing_config.yml"), "w") as file:
        yaml.safe_dump(config, file)

    # --------------
    # Perform pre-processing
    # --------------
    num_datasets = len(config["datasets"])
    try:
        for i, (dataset_name, preprocessing_kwargs) in enumerate(config["datasets"].items()):
            # Maybe print the dataset we will preprocess and save
            if verbose:
                print(f"Preprocessing and saving dataset {i + 1}/{num_datasets}: '{dataset_name}'...")

            # Save the data with the preprocessing specifications
            get_dataset(dataset_name).save_eeg_as_numpy_arrays(path=path, **preprocessing_kwargs, **config["general"])
    except DataError:
        print("Saving to numpy failed...")
        shutil.rmtree(path)


if __name__ == "__main__":
    main()
