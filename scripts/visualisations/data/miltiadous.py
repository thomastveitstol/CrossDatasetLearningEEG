"""
Script for plotting the preprocessed version of the Miltiadous

Conclusion:
    Looks quite good, although not sure if all is resting state (see e.g. subject 56).
"""
import os

import mne
from matplotlib import pyplot

from cdl_eeg.data.paths import get_raw_data_storage_path


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject = 24
    derivatives = True  # boolean, indicating cleaned/not cleaned data

    # Preprocessing
    l_freq = 1
    h_freq = 45

    # -----------------
    # Load data
    # -----------------
    # Make path
    dataset_path = os.path.join(get_raw_data_storage_path(), "miltiadous")
    dataset_path = os.path.join(dataset_path, "derivatives") if derivatives else dataset_path

    subject_id = tuple(dataset for dataset in os.listdir(dataset_path) if dataset[:4] == "sub-")[subject]
    path = os.path.join(dataset_path, subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set")

    # Make MNE raw object
    eeg: mne.io.eeglab.eeglab.RawEEGLAB = mne.io.read_raw_eeglab(input_fname=path, preload=True)

    # -----------------
    # Some additional preprocessing steps
    # -----------------
    # Filtering
    eeg.filter(l_freq=l_freq, h_freq=h_freq)

    # Re-referencing
    eeg.set_eeg_reference(ref_channels="average")

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd(fmax=h_freq+15).plot()

    pyplot.show()


if __name__ == "__main__":
    main()
