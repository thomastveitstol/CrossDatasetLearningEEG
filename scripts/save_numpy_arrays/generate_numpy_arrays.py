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
from cdl_eeg.data.preprocessing import create_folder_name


def deprecated_main():
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
            get_dataset(dataset_name).save_eeg_as_numpy_arrays(
                path=path, subject_ids=None, **preprocessing_kwargs, **config["general"]
            )
    except DataError:
        print("Saving to numpy failed...")
        shutil.rmtree(path)


def main():
    verbose = True

    # --------------
    # Load config file for preprocessing
    # --------------
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --------------
    # Create path and folders
    # --------------
    # Create main folder
    path = os.path.join(get_numpy_data_storage_path(),
                        f"preprocessed_{date.today()}_{datetime.now().strftime('%H%M%S')}")
    os.mkdir(path)

    # Save the config file to path
    shutil.copy(src=config_path, dst=path)

    # Create sub folders
    for freq_band in config["frequency_bands"]:
        l_freq, h_freq = freq_band
        for apply_autoreject in (True, False):
            for resample_multiple in config["resample_fmax_multiples"]:
                sub_folder = os.path.join(path, create_folder_name(l_freq=l_freq, h_freq=h_freq,
                                                                   is_autorejected=apply_autoreject,
                                                                   resample_multiple=resample_multiple))
                os.mkdir(sub_folder)

                # Make folder for each dataset
                for dataset_name in config["Datasets"]:
                    dataset_array_folder = os.path.join(sub_folder, dataset_name)
                    os.mkdir(dataset_array_folder)

    # --------------
    # Perform pre-processing
    # --------------
    num_datasets = len(config["Datasets"])
    for i, (dataset_name, preprocessing_kwargs) in enumerate(config["Datasets"].items()):
        # Maybe print the dataset we will preprocess and save
        if verbose:
            print(f"Preprocessing and saving dataset {i + 1}/{num_datasets}: '{dataset_name}'...")

        # Save the data with the preprocessing specifications
        get_dataset(dataset_name).save_epochs_as_numpy_arrays(
            path=dataset_array_folder, frequency_bands=config["frequency_bands"],
            resample_fmax_multiples=config["resample_fmax_multiples"], **preprocessing_kwargs, **config["general"],
            plot_data=False
        )


if __name__ == "__main__":
    main()
