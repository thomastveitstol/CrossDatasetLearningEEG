import os.path

import mne.io

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
        # Extract session (should be 't1' or 't2'). 't2' is not available for all subjects
        session = kwargs["session"]
        assert session in ("t1", "t2"), f"Expected session to be either 't1' or 't2', but found {session}"

        # Create path
        subject_path = os.path.join("derivatives", "cleaned_data", subject_id, f"ses-{session}", "eeg",
                                    f"{subject_id}_ses-{session}_task-resteyesc_eeg.edf")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Load MNE object and return
        return mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)
