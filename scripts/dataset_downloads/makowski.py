"""
Script for downloading the dataset provided at https://openneuro.org/datasets/ds004511/versions/1.0.2

References:
    Makowski, Dominique and Pham, Tam and Lau, Zen Juen (2023). Deception_data. OpenNeuro. [Dataset] doi:
        doi:10.18112/openneuro.ds004511.v1.0.2

Notes:
    According to the GitHub project, this is a side-project with no official existence

"""
import os

import openneuro

from cdl_eeg.data.paths import get_raw_data_storage_path


def main() -> None:
    # Make directory
    folder_name = "makowski"
    path = os.path.join(get_raw_data_storage_path(), folder_name)
    os.mkdir(path)

    # Download from OpenNeuro
    # todo: many unused files which are downloaded now (and 202GB of memory). Better exclude non-resting state
    openneuro.download(dataset="ds004511", target_dir=path)


if __name__ == "__main__":
    main()
