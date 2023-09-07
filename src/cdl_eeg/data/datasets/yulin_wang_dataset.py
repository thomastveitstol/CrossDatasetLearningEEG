import os
import pathlib

import mne
import pandas
from pymatreader import read_mat

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase


class YulinWang(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> YulinWang().name
    'yulin_wang'
    """

    __slots__ = ()

    def __init__(self):
        super().__init__()

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

    # ----------------
    # Methods for channel system
    # ----------------
    def _get_electrode_positions(self, subject_id=None):
        # todo: does not contain CPz

        # Create path to .tsv file  todo: hard-coding session 1 and recording
        subject_path = f"{subject_id}/ses-session1/eeg/{subject_id}_ses-session1_electrodes.tsv"
        path = os.path.join(self.get_mne_path(), subject_path)

        # Load .tsv file
        df = pandas.read_csv(path, delimiter="\t")

        # Extract channel names and coordinates
        ch_names = df["name"]
        x_vals = df["x"]
        y_vals = df["y"]
        z_vals = df["z"]

        # Make it a dict and return it
        return {ch_name: (x, y, z) for ch_name, x, y, z in zip(ch_names, x_vals, y_vals, z_vals)}

    def _get_template_electrode_positions(self):
        # Using the positions from the chanlocs62.mat file in derivatives folder
        path = os.path.join(self.get_mne_path(), "derivatives", "preprocessed data", "chanlocs62.mat")

        # Load the file
        mat_file = read_mat(path)

        # Extract positions
        ch_positions = mat_file["chanlocs"]

        ch_names = ch_positions["labels"]
        x_vals = ch_positions["X"]
        y_vals = ch_positions["Y"]
        z_vals = ch_positions["Z"]

        # Convert to dict and return
        return {ch_name: (x, y, x) for ch_name, x, y, z in zip(ch_names, x_vals, y_vals, z_vals)}

    def channel_name_to_index(self):
        # todo: make tests
        raise NotImplementedError
