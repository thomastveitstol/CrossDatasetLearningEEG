import json
import os

import mne
import numpy

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method


def _conversion_mapping():
    return {"Fp1-AVG": "Fp1", "F3-AVG": "F3", "C3-AVG": "C3", "P3-AVG": "P3", "O1-AVG": "O1", "Fp2-AVG": "Fp2",
            "F4-AVG": "F4", "C4-AVG": "C4", "P4-AVG": "P4", "O2-AVG": "O2", "F7-AVG": "F7", "T3-AVG": "T3",
            "T5-AVG": "T5", "F8-AVG": "F8", "T4-AVG": "T4", "T6-AVG": "T6", "FZ-AVG": "Fz", "CZ-AVG": "Cz",
            "PZ-AVG": "Pz"}


def _cau_name_to_standard(ch_name):
    """
    Convert from CAUEEG channel name to standard MNE channel name

    Parameters
    ----------
    ch_name : str

    Returns
    -------
    str

    Examples
    --------
    >>> _cau_name_to_standard("PZ-AVG")
    'Pz'
    >>> _cau_name_to_standard("NotAnElectrode")
    'NotAnElectrode'
    """
    # Convert channel name to MNE readable and return
    try:
        return _conversion_mapping()[ch_name]
    except KeyError:
        return ch_name


class CAUEEG(EEGDatasetBase):
    """
    This is a dataset where you need to apply for access.

    Examples:
    ----------
    >>> CAUEEG().name
    'CAUEEG'
    >>> CAUEEG.get_available_targets()
    ('age', 'alzheimers', 'dementia', 'mci', 'normal')
    >>> len(CAUEEG().get_subject_ids())
    1379
    >>> CAUEEG._channel_names
    ('Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz')
    """

    __slots__ = ()
    _channel_names = tuple(_cau_name_to_standard(name) for name in ("Fp1-AVG", "F3-AVG", "C3-AVG", "P3-AVG", "O1-AVG",
                                                                    "Fp2-AVG", "F4-AVG", "C4-AVG", "P4-AVG", "O2-AVG",
                                                                    "F7-AVG", "T3-AVG", "T5-AVG", "F8-AVG", "T4-AVG",
                                                                    "T6-AVG", "FZ-AVG", "CZ-AVG", "PZ-AVG"))
    # todo: make tests

    _montage_name = "standard_1020"  # todo: check

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, *, preload=True):
        # Create path
        path = os.path.join(self.get_mne_path(), "signal", "edf", f"{subject_id}.edf")

        # Create MNE raw object
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)

        # Drop non-eeg channels
        raw.drop_channels(("Photic", "EKG"))

        # Rename channels
        raw.rename_channels(_conversion_mapping())

        return raw

    def _get_subject_ids(self):
        """
        Get the subject IDs available. The names are derived from the available .edf files

        Returns
        -------
        tuple[str, ...]
            Subject IDs

        Examples
        --------
        >>> CAUEEG().get_subject_ids()[:5]
        ('00363', '00726', '00001', '00002', '00003')
        """
        # Get all file names
        file_names = os.listdir(os.path.join(self.get_mne_path(), "signal", "edf"))

        # Remove suffix and return
        return tuple(file_name[:-4] for file_name in file_names)

    # ----------------
    # Targets
    # ----------------
    def _get_participant_info(self):
        """
        Get participant info into a dict

        Returns
        -------
        dict[str, dict[str, typing.Any]]
            Keys are subject IDs

        Examples
        --------
        >>> CAUEEG()._get_participant_info()["00397"]
        {'age': 68, 'symptom': ['mci', 'mci_amnestic']}
        """
        # Read the .json file
        with open(os.path.join(self.get_mne_path(), "annotation.json"), "r") as file:
            data = json.load(file)["data"]

        # Convert data to dict
        participant_info = {subject["serial"]: {"age": subject["age"], "symptom": subject["symptom"]}
                            for subject in data}

        return participant_info

    @target_method
    def age(self, subject_ids):
        """
        Examples
        --------
        >>> CAUEEG().age(("00001", "00847", "00977", "01189", "00188"))
        array([78, 73, 57, 76, 88])
        """
        # Get participant info
        participant_info = self._get_participant_info()

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([participant_info[sub_id]["age"] for sub_id in subject_ids])

    @target_method
    def mci(self, subject_ids):
        """
        Examples
        --------
        >>> CAUEEG().mci(("00001", "00847", "00977", "01189", "00188"))
        [True, False, False, True, True]
        """
        # Get participant info
        participant_info = self._get_participant_info()

        # Extract the ages of the subjects, in the same order as the input argument
        return ["mci" in participant_info[sub_id]["symptom"] for sub_id in subject_ids]

    @target_method
    def dementia(self, subject_ids):
        """
        Examples
        --------
        >>> CAUEEG().dementia(("00001", "00847", "00977", "01189", "00188"))
        [False, False, False, False, False]
        """
        # Get participant info
        participant_info = self._get_participant_info()

        # Extract the ages of the subjects, in the same order as the input argument
        return ["dementia" in participant_info[sub_id]["symptom"] for sub_id in subject_ids]

    @target_method
    def alzheimers(self, subject_ids):
        """
        Examples
        --------
        >>> CAUEEG().alzheimers(("00001", "00847", "00977", "01189", "00188"))
        [False, False, False, False, False]
        """
        # Get participant info
        participant_info = self._get_participant_info()

        # Extract the ages of the subjects, in the same order as the input argument
        return ["ad" in participant_info[sub_id]["symptom"] for sub_id in subject_ids]

    @target_method
    def normal(self, subject_ids):
        """
        Examples
        --------
        >>> CAUEEG().normal(("00001", "00847", "00977", "01189", "00188"))
        [False, False, False, False, False]
        """
        # Get participant info
        participant_info = self._get_participant_info()

        # Extract the ages of the subjects, in the same order as the input argument
        return ["normal" in participant_info[sub_id]["symptom"] for sub_id in subject_ids]

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # Following the international 10-20 system  todo: where is this documented?
        montage = mne.channels.make_standard_montage(self._montage_name)
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(channel_positions[_cau_name_to_standard(ch_name)]) for ch_name in self._channel_names}

    def channel_name_to_index(self):
        # todo: make tests
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}


# ---------------
# Functions
# ---------------
def _standard_name_to_cau(ch_name):
    """
    Convert from standard channel name to CAUEEG version

    Parameters
    ----------
    ch_name : str

    Returns
    -------
    str

    Examples
    --------
    >>> _standard_name_to_cau("Pz")
    'PZ-AVG'
    >>> _standard_name_to_cau("NotAnElectrode")
    'NotAnElectrode'
    """
    # Convert channel name to CAU EEG form
    conversion = {mne_name: cau_name for cau_name, mne_name in _conversion_mapping().items()}
    try:
        return conversion[ch_name]
    except KeyError:
        return ch_name
