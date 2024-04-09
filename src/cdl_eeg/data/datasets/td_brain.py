import os
from typing import Tuple, List

import mne
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase


class TDBrain(EEGDatasetBase):
    """
    The TDBRAIN dataset

    Paper:
        van Dijk, H., van Wingen, G., Denys, D. et al. The two decades brainclinics research archive for insights in
        neurophysiology (TDBRAIN) database. Sci Data 9, 333 (2022). https://doi.org/10.1038/s41597-022-01409-z
    Link: www.brainclinics.com/resources

    Examples
    --------
    >>> len(TDBrain().get_subject_ids())
    1274
    >>> TDBrain().get_subject_ids()[:3]
    ('sub-19681349', 'sub-19681385', 'sub-19684666')
    """

    # Extracting channel names from Table 3 in the paper
    # TODO: check ordering
    _channel_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
                      "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")
    _montage_name = "standard_1020"  # 10-10 according to the paper

    def get_subject_ids(self) -> Tuple[str, ...]:
        """Get the subject IDs available. Have to override due to (1) a minor variation in column name, and (2) repeated
        subject IDs"""
        # Get the subject IDs from participants file
        participants = tuple(pandas.read_csv(self.get_participants_tsv_path(), sep="\t")["participants_ID"])

        # Keep only a unique set. This implementation is reproducible
        uniques: List[str] = []
        for participant in participants:
            if participant not in uniques:
                uniques.append(participant)

        return tuple(uniques)

    def get_participants_tsv_path(self):
        return os.path.join(self.get_mne_path(), "TDBRAIN_participants_V2_data", "TDBRAIN_participants_V2.tsv")

    def _load_single_raw_mne_object(self, subject_id):
        # Create path
        # todo: hard-coding eyes closed here...
        subject_path = f"{subject_id}/ses-1/eeg/{subject_id}_ses-1_task-restEC_eeg.vhdr"
        path = os.path.join(self.get_mne_path(), subject_path)

        # Make MNE raw object
        raw: mne.io.Raw = mne.io.read_raw_brainvision(vhdr_fname=path, preload=True, verbose=False)

        # Drop non-eeg channels
        raw.drop_channels(("VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass"))

        return raw

    # ----------------
    # Target methods
    # ----------------
    # todo: add targets. Also, when adding sex, need to check what 0 and 1 is

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
