"""
Script for downloading the dataset provided at https://openneuro.org/datasets/ds003775/versions/1.2.1

References:
    Hatlestad-Hall, C., Rygvold, T. W., & Andersson, S. (2022). BIDS-structured resting-state electroencephalography
        (EEG) data extracted from an experimental paradigm. Data in Brief, 45, 108647.
        https://doi.org/10.1016/j.dib.2022.108647
"""
import os

import openneuro

from cdl_eeg.data.datasets.hatlestad_hall import HatlestadHall


def main():
    # Make directory
    path = HatlestadHall().get_mne_path()
    os.mkdir(path)

    # Download from OpenNeuro
    openneuro.download(dataset="ds003775", target_dir=path)


if __name__ == "__main__":
    main()
