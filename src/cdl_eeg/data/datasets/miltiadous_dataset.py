import os

import mne

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, ChannelSystemBase


class MiltiadousChannelSystem(ChannelSystemBase):
    ...


class Miltiadous(EEGDatasetBase):

    def __init__(self):
        super().__init__(channel_system=MiltiadousChannelSystem())

    def load_single_mne_object(self, subject_id):
        # TODO: add derivatives functionality
        # Create path
        path = os.path.join(self.get_mne_path(), subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set")

        # Load MNE object and return
        return mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=False)
