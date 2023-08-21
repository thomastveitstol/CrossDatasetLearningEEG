"""
Script for plotting the preprocessed version of the Rockhill

Conclusion:
    Data looks good after filtering. The line noise was at 60Hz, setting h_freq to 45 is sufficient
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
    subject = 27  # 31 subjects in total

    # Preprocessing
    l_freq = 1
    h_freq = 45

    # -----------------
    # Load data
    # -----------------
    # Make path
    dataset_path = os.path.join(get_raw_data_storage_path(), "rockhill")
    subject_id = tuple(dataset for dataset in os.listdir(dataset_path) if dataset[:4] == "sub-")[subject]

    # todo: need to come back to this...
    session = f"ses-{subject_id[4:6]}"
    if session == "ses-hc":
        path = os.path.join(dataset_path, subject_id, session, "eeg", f"{subject_id}_{session}_task-rest_eeg.bdf")
    elif session == "ses-pd":
        session = "ses-off"
        path = os.path.join(dataset_path, subject_id, session, "eeg", f"{subject_id}_{session}_task-rest_eeg.bdf")
    else:
        raise ValueError(f"Session '{session}' was not recognised")

    # Make MNE raw object
    eeg: mne.io.edf.edf.RawEDF = mne.io.read_raw_bdf(input_fname=path, preload=True)

    # -----------------
    # Some additional preprocessing steps
    # -----------------
    # Keep EEG channels only
    non_eeg_channels = [ch_name for ch_name in eeg.ch_names if ch_name[:3] == "EXG"]
    eeg = eeg.pick(picks="eeg", exclude=non_eeg_channels)

    # Filtering
    eeg.filter(l_freq=l_freq, h_freq=h_freq)

    # Re-referencing
    eeg.set_eeg_reference(ref_channels="average")

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd(fmax=h_freq+20).plot()

    pyplot.show()


if __name__ == "__main__":
    main()
