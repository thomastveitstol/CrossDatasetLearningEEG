import abc
import logging
import os
from typing import Tuple

import inflection
import numpy
import pandas

from cdl_eeg.data.paths import get_raw_data_storage_path, get_numpy_data_storage_path


# --------------------
# Convenient decorators
# --------------------
def path_method(func):
    setattr(func, "_is_path_method", True)
    return func


# --------------------
# Base classes
# --------------------
class ChannelSystemBase(abc.ABC):
    ...


class EEGDatasetBase(abc.ABC):
    """
    Base class for all datasets to be used. Classes for datasets such as ChildMindInstitute Dataset (although other
    datasets are not used in the paper, I decided it would be smart to make it scalable from the beginning), should
    inherit from this class
    """

    __slots__ = "_name", "_channel_system"

    def __init__(self, channel_system, name=None):
        """
        Initialisation method

        Parameters
        ----------
        channel_system : ChannelSystemBase
            The channel system used for the EEG dataset
        name : str, optional
            Name of the EEG dataset
        """
        self._name: str = inflection.underscore(self.__class__.__name__) if name is None else name
        self._channel_system = channel_system

    @staticmethod
    def pre_process(eeg_data, *, filtering=None, resample=None, notch_filter=None, avg_reference=False):
        """
        Method for pre-processing EEG data

        Parameters
        ----------
        eeg_data : mne.io.base.BaseRaw
        filtering : tuple[float, float], optional
        resample : float, optional
        notch_filter : float, optional
        avg_reference : bool

        Returns
        -------
        mne.io.base.BaseRaw
            The pre-processed MNE raw object
        """
        # TODO: Such shared pre processing steps is not optimal. The EEG data may e.g. contain boundary events or have
        #   unequal requirements such as line noise
        # todo: Maybe try out AutoReject and use spherical spline interpolation?
        # Resampling
        if resample is not None:
            eeg_data.resample(resample, verbose=False)

        # Filtering
        if filtering is not None:
            eeg_data.filter(*filtering, verbose=False)

        # Notch filter
        if notch_filter is not None:
            eeg_data.notch_filter(notch_filter, verbose=False)

        # Re-referencing
        if avg_reference:
            eeg_data.set_eeg_reference(ref_channels="average")

        # Return the MNE object
        return eeg_data

    @abc.abstractmethod
    def load_single_mne_object(self, subject_id):
        """
        Method for loading MNE raw object of a single subject

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        mne.io.base.BaseRaw
            MNE object of the subject
        """

    def save_eeg_as_numpy_arrays(self, subject_ids=None, *, filtering=None, resample=None, notch_filter=None,
                                 num_time_steps=None, time_series_start=None):
        """
        Method for saving data as numpy arrays

        Parameters
        ----------
        subject_ids : tuple[str, ...]
            Subject IDs to convert and save as numpy arrays
        filtering : tuple[float, float], optional
            See pre_process
        resample : float, optional
        notch_filter : float, optional
        num_time_steps : int, optional
            Length of the numpy array, with unit number of time steps
        time_series_start : int, optional
            Starting point for saving the numpy array, with unit number of time steps. Indicates the number of time
            steps to skip

        Returns
        -------
        None
        """
        subject_ids = self.get_subject_ids() if subject_ids is None else subject_ids
        # ------------------
        # Input checks
        # ------------------
        # Check if all subjects are passed only once
        if len(set(subject_ids)) != len(subject_ids):
            _num_non_unique_subjects = len(set(subject_ids)) != len(subject_ids)
            raise ValueError(f"Expected all subject IDs to be unique, but there were {_num_non_unique_subjects} "
                             f"subject IDs which were passed more than once")

        # Check if all subjects are actually available
        if not all(sub_id in subject_ids for sub_id in self.get_subject_ids()):
            _unexpected_subjects = tuple(sub_id for sub_id in self.get_subject_ids() if sub_id not in subject_ids)
            raise ValueError(f"Unexpected subject IDs (N={len(_unexpected_subjects)}): {_unexpected_subjects}")

        # ------------------
        # Prepare directory and logging
        # ------------------
        # Make directory
        path = self.get_numpy_arrays_path()
        os.mkdir(path)

        # Log the process
        logging.basicConfig(filename=os.path.join(path, f"{self.name}.log"),
                            level=logging.INFO, format='%(asctime)s :: %(levelname)s -> %(message)s')

        logging.getLogger().addHandler(logging.StreamHandler())
        logger = logging.getLogger(__name__)

        msg = f"Saving data from the '{self.name}' dataset as numpy arrays"
        logger.info(f"{'=' * (len(msg) + 12)}")
        logger.info(f"===== {msg} =====")
        logger.info("...")

        logger.info("----- Pre-processing details -----")
        logger.info(f"Re-sampling: {'Skipped' if resample is None else resample}")
        logger.info(f"Filtering: {'Skipped' if filtering is None else filtering}")
        logger.info(f"Notch-filter: {'Skipped' if notch_filter is None else notch_filter}")
        logger.info("...")

        logger.info("----- Signal cropping details -----")
        logger.info(f"Time series start [time steps]: {'Skipped' if time_series_start is None else time_series_start}")
        logger.info(f"Time series length [time steps]: {'Skipped' if num_time_steps is None else num_time_steps}")
        logger.info("...")

        # ------------------
        # Loop through all subjects
        # ------------------
        for sub_id in subject_ids:
            # Load the EEG data as MNE object
            raw = self.load_single_mne_object(subject_id=sub_id)

            # Pre-process
            raw = self.pre_process(raw, filtering=filtering, resample=resample, notch_filter=notch_filter)

            # Convert to numpy arrays
            eeg_data = raw.get_data()
            assert eeg_data.ndim == 2, (f"Expected the EEG data to have two dimensions (channels, time steps), but "
                                        f"found shape={eeg_data.shape}")

            # (Maybe crop the signal)
            if time_series_start is not None:
                eeg_data = eeg_data[:, time_series_start:]
            if num_time_steps is not None:
                eeg_data = eeg_data[:, :num_time_steps]

            # Save the EEG data as numpy arrays
            numpy.save(os.path.join(path, sub_id), arr=eeg_data)

        logger.info("===== Saving complete =====")
        logger.info("===========================")

    def get_subject_ids(self) -> Tuple[str, ...]:
        """Get the subject IDs available. Unless this method is overridden, it will collect the IDs from the
        participants.tsv file"""
        return tuple(pandas.read_csv(self.get_participants_tsv_path(), sep="\t")["participant_id"])

    @property
    def name(self) -> str:
        return self._name

    @property
    def channel_system(self):
        return self._channel_system

    # ----------------
    # Path functions
    # ----------------
    @path_method
    def get_mne_path(self):
        return os.path.join(get_raw_data_storage_path(), self.name)

    @path_method
    def get_numpy_arrays_path(self):
        return os.path.join(get_numpy_data_storage_path(), self.name)

    @path_method
    def get_participants_tsv_path(self):
        """Get the path to the participants.tsv file"""
        return os.path.join(self.get_mne_path(), "participants.tsv")
