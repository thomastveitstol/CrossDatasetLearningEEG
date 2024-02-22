import os
import pathlib

import mne
import numpy
import pandas
from pymatreader import read_mat

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method


class YulinWang(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> YulinWang().name
    'yulin_wang'
    >>> len(YulinWang().get_subject_ids())
    60
    >>> YulinWang().get_subject_ids()[:5]
    ('sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05')
    >>> my_channels = tuple(YulinWang()._get_template_electrode_positions().keys())
    >>> len(my_channels)
    62
    >>> my_channels  # doctest: +NORMALIZE_WHITESPACE
    ('Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1',
     'CP3', 'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8',
     'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
     'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2', 'FCz')
    """

    __slots__ = ()

    # ----------------
    # Methods for loading
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **kwargs):
        # Extract visit number of recording type
        visit = kwargs["visit"]
        recording = kwargs["recording"]

        # Create path
        subject_path = pathlib.Path(f"{subject_id}/ses-session{visit}/eeg/"
                                    f"{subject_id}_ses-session{visit}_task-{recording}_eeg")
        subject_path = subject_path.with_suffix(".eeg")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Make MNE raw object
        raw = mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)

        # Maybe rename channels
        if "Cpz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"Cpz": "CPz"})
        if "FPz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"FPz": "Fpz"})

        return raw

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
        raw = mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)

        # Maybe rename channels
        if "Cpz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"Cpz": "CPz"})
        if "FPz" in raw.info["ch_names"]:
            mne.rename_channels(raw.info, mapping={"FPz": "Fpz"})

        return raw

    # ----------------
    # Targets
    # ----------------
    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["age"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

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

    def _deprecated_get_template_electrode_positions(self):
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
        return {ch_name: (x, y, z) for ch_name, x, y, z in zip(ch_names, x_vals, y_vals, z_vals)}

    def _get_template_electrode_positions(self):
        # Following the international 10-20 system according to the original paper. Thus using MNE default
        montage = mne.channels.make_standard_montage("standard_1020")
        channel_positions = montage.get_positions()["ch_pos"]

        # ---------------
        # Read the channel names
        # ---------------
        # Using the positions from the chanlocs62.mat file in derivatives folder
        path = os.path.join(self.get_mne_path(), "derivatives", "preprocessed data", "chanlocs62.mat")

        # Load the file
        mat_file = read_mat(path)

        # Extract channel names
        channel_names = mat_file["chanlocs"]["labels"]

        # Correct CPz channel name (does not currently match the MNE object)
        cpz_idx = channel_names.index("Cpz")
        channel_names[cpz_idx] = "CPz"

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: channel_positions[ch_name] for ch_name in channel_names}

    def channel_name_to_index(self):
        # todo: make tests
        return {ch_name: i for i, ch_name in enumerate(self._get_template_electrode_positions())}
