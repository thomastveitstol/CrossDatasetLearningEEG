"""
Script for downloading the dataset provided at https://openneuro.org/datasets/ds004148/versions/1.0.1

todo: find out how to cite properly

Reference:
Yulin Wang and Wei Duan and Debo Dong and Lihong Ding and Xu Lei (2022). A test-retest resting and cognitive state EEG
dataset. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds004148.v1.0.1
"""
import os

import openneuro

from cdl_eeg.data.paths import get_raw_data_storage_path


def main() -> None:
    # Make directory
    folder_name = "yulin_wang"
    path = os.path.join(get_raw_data_storage_path(), folder_name)
    os.mkdir(path)

    # Download from OpenNeuro
    openneuro.download(dataset="ds004148", target_dir=path)


if __name__ == "__main__":
    main()
