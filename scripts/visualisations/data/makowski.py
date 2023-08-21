"""
Script for plotting the preprocessed version of the Makowski

Conclusions:
    A crazy amount of line noise
    Not a good dataset it seems...
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
    subject = 3
    recording = "Rest"  # Can  be one of the following: 'Rest', 'CC', 'GG'

    # Preprocessing
    l_freq = 1
    h_freq = 20

    # -----------------
    # Load data
    # -----------------
    # Make path
    dataset_path = os.path.join(get_raw_data_storage_path(), "makowski")
    subject_id = tuple(dataset for dataset in os.listdir(dataset_path) if dataset[:4] == "sub-")[subject]
    path = os.path.join(dataset_path, subject_id, "ses-01", "eeg",
                        f"{subject_id}_ses-01_task-{recording}_run-01_eeg.edf")

    # Make MNE raw object
    eeg: mne.io.edf.edf.RawEDF = mne.io.read_raw_edf(input_fname=path, preload=True)

    # -----------------
    # Some additional preprocessing steps
    # -----------------
    # Resample
    eeg.resample(sfreq=500)

    # Filtering
    eeg.notch_filter(freqs=50)
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
