"""
Script for plotting the preprocessed version of the Yulin Wang dataset, using tools from MNE.

Conclusion: The data does not look good, and needs further preprocessing
"""
import os
import pathlib

import mne
from matplotlib import pyplot

from cdl_eeg.data.paths import get_raw_data_storage_path


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject = 57  # Number between (and including) 1 and 60
    recording = "EC"  # Can  be one of the following: 'EC', 'EO', 'Ma', 'Me', 'Mu'
    visit = 3  # Number between (and including) 1 and 3

    # Preprocessing
    l_freq = 1
    h_freq = 45

    # -----------------
    # Load data
    # -----------------
    # Make path
    path_to_cleaned = "derivatives/preprocessed data/preprocessed_data"
    subject_path = pathlib.Path(f"sub{str(subject).zfill(2)}_{str(visit).zfill(2)}_{recording}")
    subject_path = subject_path.with_suffix(".set")
    path = os.path.join(get_raw_data_storage_path(), "yulin_wang", path_to_cleaned, subject_path)

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
