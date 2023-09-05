import os

import mne

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, ChannelSystemBase


class MiltiadousChannelSystem(ChannelSystemBase):
    ...


class Miltiadous(EEGDatasetBase):

    def __init__(self):
        """
        Examples:
        ----------
        >>> Miltiadous().name
        'miltiadous'
        """
        super().__init__(channel_system=MiltiadousChannelSystem())

    def _load_single_raw_mne_object(self, subject_id, **_):
        # todo: there have to be better ways of handling the input data than **_
        # Create path
        path = os.path.join(self.get_mne_path(), subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set")

        # Load MNE object and return
        return mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)

    def _load_single_cleaned_mne_object(self, subject_id, **_):
        # Create path
        path = os.path.join(self.get_mne_path(), "derivatives", subject_id, "eeg",
                            f"{subject_id}_task-eyesclosed_eeg.set")

        # Load MNE object and return
        return mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)
