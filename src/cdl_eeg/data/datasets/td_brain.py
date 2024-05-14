import os
from typing import Tuple, List

import mne
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method
from cdl_eeg.data.datasets.utils import sex_to_int


class TDBrain(EEGDatasetBase):
    """
    The TDBRAIN dataset

    Paper:
        van Dijk, H., van Wingen, G., Denys, D. et al. The two decades brainclinics research archive for insights in
        neurophysiology (TDBRAIN) database. Sci Data 9, 333 (2022). https://doi.org/10.1038/s41597-022-01409-z
    Link: www.brainclinics.com/resources

    Participant 'sub-19703068' has an error in the .eeg file in the .vhdr file. The problem is using '=' instead of "-".
    It was fixed manually.

    The .vhdr file of sub-19703550 was also fixed manually, as there were two instances where 3 was replaced by 6 in the
    ID

    Examples
    --------
    >>> len(TDBrain().get_subject_ids())
    1273
    >>> TDBrain().get_subject_ids()[:3]
    ('sub-19681349', 'sub-19681385', 'sub-19684666')
    >>> # The discrepancy in sex with the .tsv file is due to differences in integer to sex mapping
    >>> TDBrain().sex(TDBrain().get_subject_ids()[:3])
    array([1, 0, 0])
    """

    # Extracting channel names from Table 3 in the paper
    # TODO: check ordering
    _channel_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
                      "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")
    _montage_name = "standard_1020"  # 10-10 according to the paper

    def _get_subject_ids(self) -> Tuple[str, ...]:
        """Get the subject IDs available. Have to override due to (1) a minor variation in column name, and (2) repeated
        subject IDs"""
        # Get the subject IDs from participants file
        participants = tuple(pandas.read_csv(self.get_participants_tsv_path(), sep="\t")["participants_ID"])

        # Keep only a unique set. This implementation is reproducible
        uniques: List[str] = []
        for participant in participants:
            # Requiring the participant ID to be a string effectively removes a nan participant
            if participant not in uniques and isinstance(participant, str):
                uniques.append(participant)

        return tuple(uniques)

    def get_participants_tsv_path(self):
        return os.path.join(self.get_mne_path(), "TDBRAIN_participants_V2_data", "TDBRAIN_participants_V2.tsv")

    def _load_single_raw_mne_object(self, subject_id, *, preload=True):
        # Create path. We will use the first available one
        # todo: hard-coding eyes closed here...
        for session in ("ses-1", "ses-2", "ses-3"):
            subject_path = f"{subject_id}/{session}/eeg/{subject_id}_{session}_task-restEC_eeg.vhdr"
            path = os.path.join(self.get_mne_path(), subject_path)

            if os.path.isfile(path):
                break

        else:  # This will only run if we did not break after the for-loop
            raise ValueError(f"No sessions found for subject {subject_id}")

        # Make MNE raw object
        raw: mne.io.Raw = mne.io.read_raw_brainvision(vhdr_fname=path, preload=preload, verbose=False)

        # Drop non-eeg channels
        raw.drop_channels(("VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass"))

        return raw

    # ----------------
    # Target methods
    # ----------------
    @target_method
    def sex(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to sex dict. Duplicates in the .tsv file should not be a problem as the kay/value pair will just be
        # overridden. Need to remove the one nan value, thus requiring string participant IDs
        sex = {sub_id: sex_to_int(_int_to_sex(gender)) for sub_id, gender in zip(df["participants_ID"], df["gender"])
               if isinstance(sub_id, str)}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([sex[sub_id] for sub_id in subject_ids])

    @target_method
    def age(self, subject_ids):
        """
        Examples
        --------
        >>> TDBrain().age(("sub-19681349", "sub-19681385", "sub-19684666"))
        array([51.59, 49.96, 47.05])
        >>> my_subjects = TDBrain().get_subject_ids()
        >>> all(isinstance(age, float) for age in TDBrain().age(my_subjects))  # type: ignore[attr-defined]
        True
        >>> sum(tuple(numpy.isnan(age) for age in TDBrain().age(my_subjects)))  # type: ignore[attr-defined]
        """
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to age dict
        age_mapping = {sub_id: float(age.replace(",", ".")) if isinstance(age, str) else age for sub_id, age in
                       zip(df["participants_ID"], df["age"]) if isinstance(sub_id, str)}

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([age_mapping[sub_id] for sub_id in subject_ids])

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        montage = mne.channels.make_standard_montage(self._montage_name)
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}


# -------------
# Functions
# -------------
def _int_to_sex(integer):
    # This mapping was taken from the TDBRIAN_participants_V2.json file
    if integer == 0:
        return "female"
    elif integer == 1:
        return "male"
    else:
        raise ValueError(f"Unexpected integer sex: {integer}")
