import os

import mne

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase


class Miltiadous(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> Miltiadous().name
    'miltiadous'
    """

    __slots__ = ()

    _channel_names = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
                      "Fz", "Cz", "Pz")

    def __init__(self):
        super().__init__()

    # ----------------
    # Methods for loading
    # ----------------
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

    # ----------------
    # Methods for channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # Following the international 10-20 system according to the README file. Thus using MNE default
        montage = mne.channels.make_standard_montage("standard_1020")
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        # todo: make tests based on .tsv files, as I only checked a single subject for channel names and ordering
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}
