import os

import mne
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase


class HatlestadHall(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> HatlestadHall().name
    'hatlestad_hall'
    """

    __slots__ = ()

    def __init__(self):
        super().__init__()

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **kwargs):
        # Extract session (should be 't1' or 't2'). 't2' is not available for all subjects
        session = kwargs["session"]
        assert session in ("t1", "t2"), f"Expected session to be either 't1' or 't2', but found {session}"

        # Create path
        subject_path = os.path.join(subject_id, f"ses-{session}", "eeg",
                                    f"{subject_id}_ses-{session}_task-resteyesc_eeg.edf")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Make MNE raw object
        return mne.io.read_raw_edf(path, preload=True, verbose=False)

    def _load_single_cleaned_mne_object(self, subject_id, **kwargs):
        # Load Epochs object
        epochs = self.load_single_cleaned_epochs_object(subject_id, session=kwargs["session"])

        # Concatenate in time
        # todo: check with Christoffer if there may be discontinuities
        data = epochs.get_data()
        num_epochs, channels, timesteps = data.shape
        data = numpy.reshape(numpy.transpose(data, (1, 0, 2)), (channels, num_epochs * timesteps))

        # Make MNE Raw object
        return mne.io.RawArray(data, info=epochs.info, verbose=False)

    def load_single_cleaned_epochs_object(self, subject_id, session):
        """
        Method for loading cleaned Epochs object of the subject

        Parameters
        ----------
        subject_id : str
            Subject ID
        session : str
            String indicating which session to load from. Must be either 't1' or 't2'. 't2' is not available for only
            some subjects

        Returns
        -------
        mne.Epochs
            The (cleaned) Epochs object of the subject
        """
        # Extract session (should be 't1' or 't2'). 't2' is not available for all subjects
        assert session in ("t1", "t2"), f"Expected session to be either 't1' or 't2', but found {session}"

        # Create path
        subject_path = os.path.join("derivatives", "cleaned_epochs", subject_id, f"ses-{session}", "eeg",
                                    f"{subject_id}_ses-{session}_task-resteyesc_desc-epochs_eeg.set")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Load MNE object and return
        return mne.io.read_epochs_eeglab(input_fname=path, verbose=False)

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # Create path
        path = os.path.join(self.get_mne_path(), "code", "bidsify-srm-restingstate", "chanlocs",
                            "BioSemi_SRM_template_64_locs.xyz")

        # Make pandas dataframe
        df = pandas.read_table(path, header=None, delim_whitespace=True).rename(columns={1: "x", 2: "y", 3: "z",
                                                                                         4: "ch_name"})

        # Extract the needed values
        x_vals = df["x"].to_numpy()
        y_vals = df["y"].to_numpy()
        z_vals = df["z"].to_numpy()

        ch_names = tuple(df["ch_name"])

        # Convert to dict and return
        return {ch_name: (x, y, z) for ch_name, x, y, z in zip(ch_names, x_vals, y_vals, z_vals)}

    def channel_name_to_index(self):
        # Create path
        path = os.path.join(self.get_mne_path(), "code", "bidsify-srm-restingstate", "chanlocs",
                            "BioSemi_SRM_template_64_locs.xyz")

        # Make pandas dataframe
        df = pandas.read_table(path, header=None, delim_whitespace=True).rename(columns={0: "idx", 4: "ch_name"})

        # Extract the needed values
        indices = df["idx"].to_numpy() - 1  # Need to subtract 1 due to python 0 zero indexing
        ch_names = tuple(df["ch_name"])

        # Convert to dict and return
        return {ch_name: idx for ch_name, idx in zip(ch_names, indices)}
