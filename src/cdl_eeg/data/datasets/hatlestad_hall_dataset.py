import os.path

import mne
import numpy

from cdl_eeg.data.datasets.dataset_base import ChannelSystemBase, EEGDatasetBase


class HatlestadHallChannelSystem(ChannelSystemBase):
    ...


class HatlestadHall(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> HatlestadHall().name
    'hatlestad_hall'
    """

    __slots__ = ()

    def __init__(self):
        super().__init__(channel_system=HatlestadHallChannelSystem())

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
