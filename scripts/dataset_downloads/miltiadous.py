"""
Script for downloading the dataset provided at https://openneuro.org/datasets/ds004504/versions/1.0.6

References:
    Andreas Miltiadous and Katerina D. Tzimourta and Theodora Afrantou and Panagiotis Ioannidis and Nikolaos
        Grigoriadis and Dimitrios G. Tsalikakis and Pantelis Angelidis and Markos G. Tsipouras and Evripidis Glavas and
        Nikolaos Giannakeas and Alexandros T. Tzallas (2023). A dataset of EEG recordings from: Alzheimer's disease,
        Frontotemporal dementia and Healthy subjects. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds004504.v1.0.6
"""
import os

import openneuro

from cdl_eeg.data.paths import get_raw_data_storage_path


def main():
    # Make directory
    folder_name = "miltiadous"
    path = os.path.join(get_raw_data_storage_path(), folder_name)
    os.mkdir(path)

    # Download from OpenNeuro
    openneuro.download(dataset="ds004504", target_dir=path)


if __name__ == "__main__":
    main()
