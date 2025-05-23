import os

import mne.io
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method
from cdl_eeg.data.datasets.utils import sex_to_int


class OUS(EEGDatasetBase):
    """
    This is a private dataset provided by Oslo University Hospital. It was not used in the paper.

    Examples:
    ----------
    >>> OUS().name
    'ous'
    >>> OUS.get_available_targets()
    ('age', 'sex')
    >>> len(OUS().get_subject_ids())
    4811
    """

    __slots__ = ()
    _channel_names = ('Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                      'Fz', 'Cz', 'Pz', 'F9', 'F10', 'T9', 'T10', 'P9', 'P10')
    _montage_name = "standard_1020"

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **_):
        # Create path
        path = os.path.join(self.get_mne_path(), subject_id, "eeg", f"{subject_id}_task-clinical_desc-cleaned_eeg.vhdr")

        # Make MNE raw object and return
        return mne.io.read_raw_brainvision(path, preload=True, verbose=False)

    # ----------------
    # Targets
    # ----------------
    @target_method
    def sex(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: sex_to_int(sex) for name, sex in zip(df["participant_id"], df["sex"])}

        # Extract the sexes of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["age"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # Following the international 10-20 system
        montage = mne.channels.make_standard_montage("standard_1020")
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}
