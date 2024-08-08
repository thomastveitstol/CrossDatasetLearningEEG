import os

import mne
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method
from cdl_eeg.data.datasets.utils import sex_to_int


class Miltiadous(EEGDatasetBase):
    """
    Dataset from 'A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects'

    Paper:
        Miltiadous A, Tzimourta KD, Afrantou T, Ioannidis P, Grigoriadis N, Tsalikakis DG, Angelidis P, Tsipouras MG,
        Glavas E, Giannakeas N, et al. A Dataset of Scalp EEG Recordings of Alzheimer’s Disease, Frontotemporal Dementia
        and Healthy Subjects from Routine EEG. Data. 2023; 8(6):95. https://doi.org/10.3390/data8060095

    OpenNeuro:
        Andreas Miltiadous and Katerina D. Tzimourta and Theodora Afrantou and Panagiotis Ioannidis and Nikolaos
        Grigoriadis and Dimitrios G. Tsalikakis and Pantelis Angelidis and Markos G. Tsipouras and Evripidis Glavas and
        Nikolaos Giannakeas and Alexandros T. Tzallas (2024). A dataset of EEG recordings from: Alzheimer's disease,
        Frontotemporal dementia and Healthy subjects. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds004504.v1.0.7

    Examples:
    ----------
    >>> Miltiadous().name
    'Miltiadous'
    >>> Miltiadous.get_available_targets()
    ('age', 'mmse', 'sex')
    >>> len(Miltiadous().get_subject_ids())
    88
    >>> len(Miltiadous().channel_name_to_index())
    19
    """

    __slots__ = ()

    _channel_names = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
                      "Fz", "Cz", "Pz")
    _montage_name = "standard_1020"

    # ----------------
    # Methods for loading
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, *, preload=True):
        # Create path
        path = os.path.join(self.get_mne_path(), subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set")

        # Load MNE object and return
        return mne.io.read_raw_eeglab(input_fname=path, preload=preload, verbose=False)

    def _load_single_cleaned_mne_object(self, subject_id, *, preload=True):
        # Create path
        path = os.path.join(self.get_mne_path(), "derivatives", subject_id, "eeg",
                            f"{subject_id}_task-eyesclosed_eeg.set")

        # Load MNE object and return
        return mne.io.read_raw_eeglab(input_fname=path, preload=preload, verbose=False)

    # ----------------
    # Targets
    # ----------------
    @target_method
    def sex(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: sex_to_int(sex) for name, sex in zip(df["participant_id"], df["Gender"])}

        # Extract the sexes of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["Age"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def mmse(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["MMSE"])}

        # Extract the MMSE score of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

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
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}
