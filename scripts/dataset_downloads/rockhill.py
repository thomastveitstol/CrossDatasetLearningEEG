"""
Script for downloading the dataset provided at https://openneuro.org/datasets/ds002778/versions/1.0.5

References:
    Alexander P. Rockhill and Nicko Jackson and Jobi George and Adam Aron and Nicole C. Swann (2021). UC San Diego
    Resting State EEG Data from Patients with Parkinson's Disease. OpenNeuro. [Dataset] doi:
    doi:10.18112/openneuro.ds002778.v1.0.5

Notes:
    The authors want additional acknowledgement
"""
import os

import openneuro

from cdl_eeg.data.paths import get_raw_data_storage_path


def main() -> None:
    # Make directory
    folder_name = "rockhill"
    path = os.path.join(get_raw_data_storage_path(), folder_name)
    os.mkdir(path)

    # Download from OpenNeuro
    openneuro.download(dataset="ds002778", target_dir=path)


if __name__ == "__main__":
    main()
