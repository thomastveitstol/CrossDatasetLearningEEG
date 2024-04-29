import os

import mne
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method
from cdl_eeg.data.datasets.utils import sex_to_int


class Miltiadous(EEGDatasetBase):
    """
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
        # todo: make tests based on .tsv files, as I only checked a single subject for channel names and ordering
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}


# -------------
# Functions
#
# I stopped with the implementation as I don't think I want to use it
# -------------
def _merge_time_segments(_):
    raise NotImplementedError


def _get_longest_segment(onsets, durations, additional_duration, full_duration):
    """
    Function for getting the t_min and t_max for the longest segment without boundary events

    Parameters
    ----------
    onsets : numpy.ndarray
        In seconds
    durations : numpy.ndarray
        In seconds
    additional_duration : float
        Add longer time duration in seconds to all boundary events

    Returns
    -------
    tuple[float, float]
        t_max and t_min

    Examples
    --------
    >>> my_onsets = [6, 21, 40, 47]
    >>> my_durations = [4, 6, 2, 8]
    >>> _get_longest_segment(onsets=numpy.array(my_onsets), durations=numpy.array(my_durations), additional_duration=3,
    ...                      full_duration=60)
    """
    # -------------
    # Input checks
    # -------------
    if onsets.ndim != 1:
        raise ValueError(f"Expected onsets to be 1D array, but found {onsets.ndim}D")
    if not numpy.all(onsets[:-1] < onsets[1:]):
        raise ValueError(f"Expected onsets to be sorted, but found {onsets}")
    if durations.ndim != 1:
        raise ValueError(f"Expected durations to be 1D array, but found {durations.ndim}D")

    # -------------
    # Find longest time segment
    # -------------
    # If no boundary events, all data can be used
    if onsets.shape[0] == 0:
        return 0, full_duration

    # Create time segments of boundary events
    illegal_time_segments = [(onset, min(onset + dur + additional_duration, full_duration)) for onset, dur
                             in zip(onsets, durations)]

    # This does not work, must merge illegal time segments here if proceeding with this...

    # Create legal time segments
    legal_time_segments = []
    curr_start = 0
    for illegal in illegal_time_segments:
        legal_time_segments.append((curr_start, illegal[0]))
        curr_start = illegal[1]

    # Maybe add the edge
    if illegal_time_segments[-1][1] < full_duration:
        legal_time_segments.append((illegal_time_segments[-1][1], full_duration))

    return legal_time_segments, illegal_time_segments
