import os
import pathlib

import mne

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, ChannelSystemBase


class YulinWangChannelSystem(ChannelSystemBase):
    ...


class YulinWang(EEGDatasetBase):

    def __init__(self):
        super().__init__(channel_system=YulinWangChannelSystem())

    # ----------------
    # Methods for loading
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **kwargs):
        # Extract visit number of recording type
        visit = kwargs["visit"]
        recording = kwargs["recording"]

        # Create path
        subject_path = pathlib.Path(f"sub-{str(subject_id).zfill(2)}/ses-session{visit}/eeg/"
                                    f"sub-{str(subject_id).zfill(2)}_ses-session{visit}_task-{recording}_eeg")
        subject_path = subject_path.with_suffix(".eeg")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Make MNE raw object
        return mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)

    def _load_single_cleaned_mne_object(self, subject_id, **kwargs):
        # Extract visit number of recording type
        visit = kwargs["visit"]
        recording = kwargs["recording"]

        # Create path
        path_to_cleaned = "derivatives/preprocessed data/preprocessed_data"
        subject_path = pathlib.Path(f"{str(subject_id).zfill(2).replace('-', '')}_{str(visit).zfill(2)}_{recording}")
        subject_path = subject_path.with_suffix(".set")
        path = os.path.join(self.get_mne_path(), path_to_cleaned, subject_path)

        # Make MNE raw object
        return mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)
