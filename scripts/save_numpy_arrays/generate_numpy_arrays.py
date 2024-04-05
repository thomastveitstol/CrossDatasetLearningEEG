"""
Script for generating numpy arrays from the EEG data
"""
import os
import shutil
from datetime import date, datetime

import yaml

from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_numpy_data_storage_path
from cdl_eeg.data.preprocessing import create_folder_name


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
            path=path, frequency_bands=config["frequency_bands"],
            resample_fmax_multiples=config["resample_fmax_multiples"], **preprocessing_kwargs, **config["general"],
            plot_data=False
        )


if __name__ == "__main__":
    main()
