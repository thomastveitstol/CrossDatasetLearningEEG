import abc
import dataclasses
import logging
import os
from typing import Dict, List, Tuple

import enlighten
import inflection
import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.paths import get_raw_data_storage_path, get_numpy_data_storage_path
from cdl_eeg.models.region_based_pooling.utils import Electrodes3D


# --------------------
# Convenient decorators
# --------------------
def path_method(func):
    setattr(func, "_is_path_method", True)
    return func


# --------------------
# Classes
# --------------------
@dataclasses.dataclass(frozen=True)
class ChannelSystem:
    """Data class for channel systems"""
    name: str  # Should ideally be the same as dataset name
    channel_name_to_index: Dict[str, int]
    electrode_positions: Electrodes3D


class EEGDatasetBase(abc.ABC):
    """
    Base class for all datasets to be used

    todo: use Electrodes3D more
    """

    __slots__ = "_name"

    def __init__(self, name=None):
        """
        Initialisation method

        Parameters
        ----------
        name : str, optional
            Name of the EEG dataset
        """
        self._name: str = inflection.underscore(self.__class__.__name__) if name is None else name

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
            eeg_data.set_eeg_reference(ref_channels="average", verbose=False)

        # Return the MNE object
        return eeg_data

    # ----------------
    # Loading methods
    # ----------------
    def load_single_mne_object(self, subject_id, derivatives=False, **kwargs):
        """
        Method for loading MNE raw object of a single subject

        Parameters
        ----------
        subject_id : str
            Subject ID
        derivatives : bool
            For datasets where an already cleaned version is available. If True, the cleaned version will be used,
            otherwise the non-cleaned data is loaded

        Returns
        -------
        mne.io.base.BaseRaw
            MNE object of the subject
        """
        return self._load_single_cleaned_mne_object(subject_id, **kwargs) if derivatives \
            else self._load_single_raw_mne_object(subject_id, **kwargs)

    @abc.abstractmethod
    def _load_single_raw_mne_object(self, subject_id, **kwargs):
        """
        Method for loading raw data

        Parameters
        ----------
        subject_id : str
            Subject ID
        Returns
        -------
        mne.io.base.BaseRaw
            MNE object of the subject
        """

    def _load_single_cleaned_mne_object(self, subject_id, **kwargs):
        """
        Method for loading existing pre-processed data (only relevant for some datasets)

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        mne.io.base.BaseRaw
            MNE object of the subject
        """
        raise NotImplementedError("A cleaned version is not available for this class.")

    def load_numpy_arrays(self, subject_ids=None, *, time_series_start=None, num_time_steps=None):
        """
        Method for loading numpy arrays

        Parameters
        ----------
        subject_ids : tuple[str, ...]
        time_series_start : int, optional
        num_time_steps : int, optional

        Returns
        -------
        numpy.ndarray
        """
        subject_ids = self.get_subject_ids() if subject_ids is None else subject_ids

        # ------------------
        # Input checks todo: copied from save as numpy
        # ------------------
        # Check if all subjects are passed only once
        if len(set(subject_ids)) != len(subject_ids):
            _num_non_unique_subjects = len(set(subject_ids)) != len(subject_ids)
            raise ValueError(f"Expected all subject IDs to be unique, but there were {_num_non_unique_subjects} "
                             f"subject IDs which were passed more than once")

        # Check if all subjects are actually available
        if not all(sub_id in subject_ids for sub_id in self.get_subject_ids()):
            _unexpected_subjects = tuple(sub_id for sub_id in self.get_subject_ids() if sub_id not in subject_ids)
            raise ValueError(f"Unexpected subject IDs for class '{type(self).__name__}' "
                             f"(N={len(_unexpected_subjects)}): {_unexpected_subjects}")

        # ------------------
        # Loop through all subjects
        # ------------------
        # Set counter
        pbar = enlighten.Counter(total=len(subject_ids), desc="Loading", unit="subjects")

        data: List[numpy.ndarray] = []
        for sub_id in subject_ids:
            # Load the numpy array
            eeg_data = numpy.load(os.path.join(self.get_numpy_arrays_path(), f"{sub_id}.npy"))

            # (Maybe crop the signal)
            if time_series_start is not None:
                eeg_data = eeg_data[..., time_series_start:]
            if num_time_steps is not None:
                eeg_data = eeg_data[..., :num_time_steps]

            # Add the data
            data.append(numpy.expand_dims(eeg_data, axis=0))
            pbar.update()

        # Concatenate to a single numpy ndarray
        return numpy.concatenate(data, axis=0)

    def save_eeg_as_numpy_arrays(self, subject_ids=None, *, filtering=None, resample=None, notch_filter=None,
                                 avg_reference=False, num_time_steps=None, time_series_start=None, derivatives=False,
                                 **kwargs):
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
        avg_reference : bool
        num_time_steps : int, optional
            Length of the numpy array, with unit number of time steps
        time_series_start : int, optional
            Starting point for saving the numpy array, with unit number of time steps. Indicates the number of time
            steps to skip
        derivatives : bool
            For datasets where an already cleaned version is available. If True, the cleaned version will be used,
            otherwise the non-cleaned data is loaded
        kwargs
            Keyword arguments, which will be passed to load_single_mne_object

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
        logging.basicConfig(filename=os.path.join(path, f"{self.name}.log"), level=logging.INFO,
                            format='%(asctime)s :: %(levelname)s -> %(message)s')

        logging.getLogger().addHandler(logging.StreamHandler())
        logger = logging.getLogger(__name__)

        msg = f"Saving data from the '{self.name}' dataset as numpy arrays"
        logger.info(f"{'=' * (len(msg) + 12)}")
        logger.info(f"===== {msg} =====")
        logger.info("...")

        logger.info("----- Pre-processing details -----")
        logger.info(f"Derivatives: {derivatives}")
        logger.info(f"Re-sampling: {'Skipped' if resample is None else resample}")
        logger.info(f"Filtering: {'Skipped' if filtering is None else filtering}")
        logger.info(f"Notch-filter: {'Skipped' if notch_filter is None else notch_filter}")
        logger.info(f"Average referencing: {avg_reference}")
        logger.info("...")

        logger.info("----- Signal cropping details -----")
        logger.info(f"Time series start [time steps]: {'Skipped' if time_series_start is None else time_series_start}")
        logger.info(f"Time series length [time steps]: {'Skipped' if num_time_steps is None else num_time_steps}")
        logger.info("...")

        logger.info("----- Additional keyword arguments -----")
        if kwargs:
            for key, value in kwargs.items():
                logger.info(f"Argument '{key}': {value}")
        else:
            logger.info("No additional keyword arguments were passed")
        logger.info("...")

        # ------------------
        # Loop through all subjects
        # ------------------
        for sub_id in subject_ids:
            # Load the EEG data as MNE object
            raw = self.load_single_mne_object(subject_id=sub_id, derivatives=derivatives, **kwargs)

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

    # ----------------
    # Properties
    # ----------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def channel_system(self) -> ChannelSystem:
        return ChannelSystem(name=self.name, channel_name_to_index=self.channel_name_to_index(),
                             electrode_positions=Electrodes3D(self.get_electrode_positions()))  # type: ignore[arg-type]

    # ----------------
    # Path methods
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

    # ----------------
    # Channel system
    # ----------------
    def get_electrode_positions(self, subject_id=None):
        """
        Method for getting the electrode positions (cartesian coordinates) of a specified subject. If this method is not
        overridden or None is passed, a template is used instead
        Parameters
        ----------
        subject_id : str, optional
            Subject ID

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Cartesian coordinates of the channels. Keys are channel names
        """
        if subject_id is None:
            return self._get_template_electrode_positions()
        else:
            # Use subject specific coordinates. If not implemented, use template instead
            try:
                return self._get_electrode_positions(subject_id)
            except NotImplementedError:
                # todo: consider raising a warning here
                return self._get_template_electrode_positions()

    def _get_electrode_positions(self, subject_id=None):
        """
        Method for getting the electrode positions (cartesian coordinates) of a specified subject.

        Parameters
        ----------
        subject_id : str, optional
            Subject ID

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Cartesian coordinates of the channels. Keys are channel names
        """
        raise NotImplementedError

    def _get_template_electrode_positions(self):
        """
        Method for getting the template electrode positions (cartesian coordinates)

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Cartesian coordinates of the channels. Keys are channel names
        """
        raise NotImplementedError

    @abc.abstractmethod
    def channel_name_to_index(self):
        """
        Get the mapping from channel name to index

        Returns
        -------
        dict[str, int]
            Keys are channel name, value is the row-position in the data matrix
        """

    def plot_electrode_positions(self, subject_id=None, annotate=True):
        """
        Method for 3D plotting the electrode positions.

        Parameters
        ----------
        subject_id : str, optional
            Subject ID
        annotate : bool
            To annotate the points with channel names (True) or not (False)

        Returns
        -------
        None
        """
        # Get electrode positions
        electrode_positions = self.get_electrode_positions(subject_id=subject_id)  # todo: mypy thinks this is a float

        # Extract coordinates
        channel_names = []
        x_vals = []
        y_vals = []
        z_vals = []
        for ch_name, (x, y, z) in electrode_positions.items():  # type: ignore
            channel_names.append(ch_name)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

        # Make new figure
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot
        ax.scatter(x_vals, y_vals, z_vals)

        # Annotate the channels with channel names (if desired)
        if annotate:
            for x, y, z, channel in zip(x_vals, y_vals, z_vals, channel_names):
                ax.text(x=x, y=y, z=z, s=channel)


# ------------------
# Functions
# ------------------
def channel_names_to_indices(ch_names, channel_name_to_index):
    """
    Same as channel_name_to_index, but now you can pass in a tuple of channel names

    Parameters
    ----------
    ch_names : tuple[str, ...]
        Channel names to be mapped to indices
    channel_name_to_index : dict[str, int]
        Mapping from channel name (keys) to index in data matrix (values)

    Returns
    -------
    tuple[int, ...]
        The indices of the channel names, in the same order as ch_names

    Examples
    --------
    >>> channel_names_to_indices(("A", "C", "B", "E"), channel_name_to_index={"A": 0, "B": 1, "C": 2, "D": 3, "E": 4})
    (0, 2, 1, 4)
    """
    return tuple(channel_name_to_index[channel_name] for channel_name in ch_names)
