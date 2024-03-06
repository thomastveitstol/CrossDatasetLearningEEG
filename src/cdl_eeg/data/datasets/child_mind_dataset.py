import os
from typing import Tuple

import mne
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method


class ChildMind(EEGDatasetBase):
    """
    Dataset for the Child Mind Institute resting state data, as cleaned by Christoffer Hatlestad-Hall

    Examples
    --------
    >>> ChildMind().name
    'child_mind'
    >>> ChildMind.get_available_targets()
    ('age', 'sex')
    >>> len(ChildMind().get_subject_ids())
    2552
    """

    # ----------------
    # Loading methods
    # ----------------
    def get_subject_ids(self) -> Tuple[str, ...]:
        # Get the subject IDs from participants file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")
        participants = df["participant_id"]

        # The sex and age must be known
        sexes = df["Sex"]
        ages = df["Age"]
        participants = tuple(participant for participant, sex, age in zip(participants, sexes, ages)
                             if not (numpy.isnan(sex) or numpy.isnan(age)))

        # Keep only the ones in the preprocessed EEG data
        _eeg_availables = os.listdir(self.get_mne_path())

        return tuple(participant for participant in participants if f"{participant}.set" in _eeg_availables)

    def get_mne_path(self):
        # todo: I want to move the data to where all other datasets are, but it's a little time consuming
        return "/media/thomas/AI-Mind - Anonymised data/child_mind_data_resting_state_preprocessed"

    def _load_single_raw_mne_object(self, subject_id):
        # Create path
        path = os.path.join(self.get_mne_path(), f"{subject_id}.set")

        # Return MNE object
        return mne.io.read_raw_eeglab(path, verbose=False, preload=True)

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self, subject_id=None):
        # Following the Hydrocel system
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        channel_positions = montage.get_positions()["ch_pos"]

        return channel_positions

    def channel_name_to_index(self):
        """
        Examples
        --------
        >>> my_expected_keys = [f"E{i+1}" for i in range(128)]  # type: ignore[attr-defined]
        >>> my_expected_keys.append("Cz")
        >>> list(ChildMind().channel_name_to_index().keys()) == my_expected_keys
        True
        >>> list(ChildMind().channel_name_to_index().values()) == [i for i in range(129)]  # type: ignore[attr-defined]
        True
        """
        channel_names = self.get_electrode_positions()
        return {channel_name: i for i, channel_name in enumerate(channel_names)}

    # ----------------
    # Target methods
    # ----------------
    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["age"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def sex(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict. 0=Male, 1=Female according to link below
        # http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/sharing_neuro.html
        sub_id_to_sex = {name: sex for name, sex in zip(df["participant_id"], df["sex"])}

        # Extract the sexes of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_sex[sub_id] for sub_id in subject_ids])
