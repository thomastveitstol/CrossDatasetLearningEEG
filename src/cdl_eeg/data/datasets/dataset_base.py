import abc
import dataclasses
import os
from typing import Dict, Tuple, List, Optional

import enlighten
import mne
import numpy
import pandas
from matplotlib import pyplot
from mne.transforms import _cart_to_sph, _pol_to_cart

from cdl_eeg.data.paths import get_raw_data_storage_path, get_numpy_data_storage_path
from cdl_eeg.data.preprocessing import save_preprocessed_epochs
from cdl_eeg.models.region_based_pooling.utils import ELECTRODES_3D


# --------------------
# Convenient decorators
# --------------------
def target_method(func):
    setattr(func, "_is_target_method", True)
    return func


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
    electrode_positions: ELECTRODES_3D
    montage_name: Optional[str] = None


class EEGDatasetBase(abc.ABC):
    """
    Base class for all datasets to be used
    """

    __slots__ = "_name"

    _montage_name: Optional[str] = None

    def __init__(self, name=None):
        """
        Initialisation method

        Parameters
        ----------
        name : str, optional
            Name of the EEG dataset
        """
        self._name: str = self.__class__.__name__ if name is None else name  # todo
        # self._name: str = inflection.underscore(self.__class__.__name__) if name is None else name

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
        # Load raw object
        raw = self._load_single_cleaned_mne_object(subject_id, **kwargs) if derivatives \
            else self._load_single_raw_mne_object(subject_id, **kwargs)

        # Set montage
        raw.set_montage(self.channel_system.montage_name)
        return raw

    @abc.abstractmethod
    def _load_single_raw_mne_object(self, *args, **kwargs):
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

    def _load_single_cleaned_mne_object(self, *args, **kwargs):
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
        raise NotImplementedError(f"A cleaned version is not available for this class ({self.__class__.__name__}).")

    def load_numpy_arrays(self, subject_ids=None, pre_processed_version=None, *, time_series_start=None,
                          num_time_steps=None, channels=None, required_target=None):
        """
        Method for loading numpy arrays

        Parameters
        ----------
        subject_ids : tuple[str, ...], optional
        pre_processed_version : str, optional
            The pre-processed version. That is, the numpy arrays should be stored inside
            os.path.join(self.get_numpy_arrays_path(), pre_processed_version)
        time_series_start : int, optional
        num_time_steps : int, optional
        channels: tuple[str, ...], optional
        required_target : str, optional

        Returns
        -------
        numpy.ndarray
        """
        # Maybe set defaults
        path = self.get_numpy_arrays_path(pre_processed_version)
        subject_ids = self.get_subject_ids() if subject_ids is None else subject_ids

        # If there is a required target availability, prune the subject IDs
        if required_target is not None:
            _accepted_subject_ids: List[str] = []
            _targets = self.load_targets(target=required_target, subject_ids=subject_ids)
            for sub_id, target in zip(subject_ids, _targets):
                if not numpy.isnan(target):
                    _accepted_subject_ids.append(sub_id)
            subject_ids = tuple(_accepted_subject_ids)

        # ------------------
        # Input checks todo: copied from save as numpy
        # ------------------
        # Check if all subjects are passed only once
        if len(set(subject_ids)) != len(subject_ids):
            _num_non_unique_subjects = len(set(subject_ids)) != len(subject_ids)
            raise ValueError(f"Expected all subject IDs to be unique, but there were {_num_non_unique_subjects} "
                             f"subject IDs which were passed more than once")

        # Check if all subjects are actually available
        available_subjects = self.get_subject_ids()
        if not all(sub_id in available_subjects for sub_id in subject_ids):
            _unexpected_subjects = tuple(sub_id for sub_id in subject_ids if sub_id not in self.get_subject_ids())
            raise ValueError(f"Unexpected subject IDs for class '{type(self).__name__}' "
                             f"(N={len(_unexpected_subjects)}): {_unexpected_subjects}")

        # ------------------
        # Loop through all subjects
        # ------------------
        # Set counter
        pbar = enlighten.Counter(total=len(subject_ids), desc="Loading", unit="subjects")

        data = []
        for sub_id in subject_ids:
            # Load the numpy array
            eeg_data = numpy.load(os.path.join(path, f"{sub_id}.npy"))

            # (Maybe) crop the signal
            if time_series_start is not None:
                eeg_data = eeg_data[..., time_series_start:]
            if num_time_steps is not None:
                eeg_data = eeg_data[..., :num_time_steps]

            # (Maybe) remove unwanted signals
            if channels is not None:
                indices = channel_names_to_indices(ch_names=channels,
                                                   channel_name_to_index=self.channel_name_to_index())
                eeg_data = eeg_data[indices]

            # Add the data
            data.append(numpy.expand_dims(eeg_data, axis=0))
            pbar.update()

        # Concatenate to a single numpy ndarray
        return numpy.concatenate(data, axis=0)

    def save_epochs_as_numpy_arrays(self, path, *, subject_ids, derivatives, excluded_channels, main_band_pass,
                                    frequency_bands, notch_filter, num_epochs, epoch_duration, epoch_overlap,
                                    time_series_start_secs, autoreject_resample, resample_fmax_multiples, seed,
                                    plot_data=False, **kwargs):
        """
        Method for saving data as numpy arrays

        Parameters
        ----------
        path : str
            Where to store the data
        subject_ids : tuple[str, ...]
            Subject IDs of which to save numpy arrays of
        derivatives : bool
            To use cleaned versions or not
        excluded_channels : tuple[str, ...]
            Channels to exclude
        main_band_pass : tuple[float, float]
            Band-pass filtering
        frequency_bands : tuple[tuple[float, float], ...]
            All frequency bands to save
        notch_filter : float, optional
            Frequency of notch filter. If None, no notch filter will be used
        num_epochs : int
            Number of epochs to use per subject
        epoch_duration : float
            Duration of each epoch in seconds
        epoch_overlap : float
            Duration of epoch overlap in seconds
        time_series_start_secs : float
            Start of the time series in seconds
        autoreject_resample : tuple[float, float], optional
            Sampling frqeuency before applying autoreject
        resample_fmax_multiples : float
            The resampling frequency will be this parameter multiplied with f_max
        seed : int
            Seed for reproducability
        plot_data : bool
            To plot the data or not (useful for debugging purposes)
        kwargs

        Returns
        -------
        None
        """
        subject_ids = self.get_subject_ids(preprocessed_version=None) if subject_ids is None else subject_ids

        # ------------------
        # Input checks
        # ------------------
        # Check if all subjects are passed only once
        if len(set(subject_ids)) != len(subject_ids):
            _num_non_unique_subjects = len(subject_ids) - len(set(subject_ids))
            raise ValueError(f"Expected all subject IDs to be unique, but there were {_num_non_unique_subjects} "
                             f"subject IDs which were passed more than once")

        # Check if all subjects are actually available
        available_subjects = self.get_subject_ids(preprocessed_version=None)
        if not all(sub_id in available_subjects for sub_id in subject_ids):
            _unexpected_subjects = tuple(sub_id for sub_id in self.get_subject_ids() if sub_id not in subject_ids)
            raise ValueError(f"Unexpected subject IDs (N={len(_unexpected_subjects)}): {_unexpected_subjects}")

        # ------------------
        # Loop through all subjects
        # ------------------
        pbar = enlighten.Counter(total=len(subject_ids), desc=type(self).__name__, unit="subjects")
        for sub_id in subject_ids:
            # Load the EEG data as MNE object
            raw = self.load_single_mne_object(subject_id=sub_id, derivatives=derivatives, **kwargs)

            # Save preprocessed versions
            save_preprocessed_epochs(
                raw, excluded_channels=excluded_channels, main_band_pass=main_band_pass,
                frequency_bands=frequency_bands, notch_filter=notch_filter, num_epochs=num_epochs,
                epoch_duration=epoch_duration, epoch_overlap=epoch_overlap,
                time_series_start_secs=time_series_start_secs, autoreject_resample=autoreject_resample,
                resample_fmax_multiples=resample_fmax_multiples, subject_id=sub_id, path=path,
                plot_data=plot_data, dataset_name=type(self).__name__, seed=seed
            )

            # Update progress bar
            pbar.update()

    def get_subject_ids(self, preprocessed_version=None) -> Tuple[str, ...]:
        """Get the subject IDs available. If a preprocessed version is specified (not None), it only returns the IDs
        which are present in that preprocessed version"""
        # Return all available ones if preprocessed version is not specified
        if preprocessed_version is None:
            return self._get_subject_ids()

        # ----------------
        # Remove all subjects which are not contained in the specified preprocessed version
        # ----------------
        subjects = self._get_subject_ids()
        path = os.path.join(get_numpy_data_storage_path(), preprocessed_version, self.name)

        # Need to remove .npy. Can't use .removesuffix(".npy") for Python 3.8 compatibility reasons
        _available_subjects = tuple(subject_npy[:-4] for subject_npy in os.listdir(path))
        return tuple(subject for subject in subjects if subject in _available_subjects)

    def _get_subject_ids(self) -> Tuple[str, ...]:
        """Get the subject IDs available. Unless this method is overridden, it will collect the IDs from the
        participants.tsv file"""
        return tuple(pandas.read_csv(self.get_participants_tsv_path(), sep="\t")["participant_id"])

    # ----------------
    # Target methods
    # ----------------
    def load_targets(self, target, subject_ids=None):
        """
        Method for loading targets

        Parameters
        ----------
        target : str
        subject_ids : tuple[str, ...]

        Returns
        -------
        numpy.ndarray
        """
        subject_ids = self.get_subject_ids() if subject_ids is None else subject_ids

        # Input check
        if target not in self.get_available_targets():
            raise ValueError(f"Target '{target}' was not recognised. Make sure that the method passed shares the name "
                             f"with the implemented method you want to use. The targets available for this class "
                             f"({type(self).__name__}) are: {self.get_available_targets()}")

        # Return the targets  todo: check if 'subject_ids' can be a required input for the decorated methods
        return getattr(self, target)(subject_ids=subject_ids)

    @classmethod
    def get_available_targets(cls):
        """Get all target methods available for the class. The target method must be decorated by @target_method to be
        properly registered"""
        # Get all target methods
        target_methods: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a target method
            if callable(attribute) and getattr(attribute, "_is_target_method", False):
                target_methods.append(method)

        # Convert to tuple and return
        return tuple(target_methods)

    # ----------------
    # Properties
    # ----------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def num_channels(self):
        return len(self.channel_name_to_index())

    @property
    def channel_system(self) -> ChannelSystem:
        return ChannelSystem(name=self.name, channel_name_to_index=self.channel_name_to_index(),
                             electrode_positions=self.get_electrode_positions(), montage_name=self._montage_name)

    # ----------------
    # Path methods
    # ----------------
    @path_method
    def get_mne_path(self):
        return os.path.join(get_raw_data_storage_path(), self.name)

    @path_method
    def get_numpy_arrays_path(self, pre_processed_version=None):
        if pre_processed_version is None:
            return os.path.join(get_numpy_data_storage_path(), self.name)
        else:
            return os.path.join(get_numpy_data_storage_path(), pre_processed_version, self.name)

    @path_method
    def get_participants_tsv_path(self):
        """Get the path to the participants.tsv file"""
        return os.path.join(self.get_mne_path(), "participants.tsv")

    # ----------------
    # Channel system  TODO: consider changing to classmethods
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
        ELECTRODES_3D
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
        ELECTRODES_3D
            Cartesian coordinates of the channels. Keys are channel names
        """
        raise NotImplementedError

    def _get_template_electrode_positions(self):
        """
        Method for getting the template electrode positions (cartesian coordinates)

        Returns
        -------
        ELECTRODES_3D
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

    def plot_electrode_positions(self, subject_id=None, annotate=True, ax=None):
        """
        Method for 3D plotting the electrode positions.

        Parameters
        ----------
        subject_id : str, optional
            Subject ID
        annotate : bool
            To annotate the points with channel names (True) or not (False)
        ax: optional

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

        # Maybe make new figure
        if ax is None:
            fig = pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot
        ax.scatter(x_vals, y_vals, z_vals)

        # Annotate the channels with channel names (if desired)
        if annotate:
            for x, y, z, channel in zip(x_vals, y_vals, z_vals, channel_names):
                ax.text(x=x, y=y, z=z, s=channel)

    def plot_2d_electrode_positions(self, subject_id=None, annotate=True):
        # Get electrode positions
        electrode_positions = self.get_electrode_positions(subject_id=subject_id)

        # Apply the same steps as _auto_topomap_coordinates from MNE.transforms
        cartesian_coords = _cart_to_sph(tuple(electrode_positions.values()))  # type: ignore
        out = _pol_to_cart(cartesian_coords[:, 1:][:, ::-1])
        out *= cartesian_coords[:, [0]] / (numpy.pi / 2.)

        # Extract coordinates
        channel_names = []
        x_vals = []
        y_vals = []
        for ch_name, (x, y) in zip(electrode_positions, out):  # type: ignore
            channel_names.append(ch_name)
            x_vals.append(x)
            y_vals.append(y)

        # Plot
        pyplot.scatter(x_vals, y_vals)

        # Annotate the channels with channel names (if desired)
        if annotate:
            for x, y, channel in zip(x_vals, y_vals, channel_names):
                pyplot.text(x=x, y=y, s=channel)


# ------------------
# Functions
# ------------------
def get_channel_name_order(channel_name_to_index):
    """
    This function ensures that one obtains the correct channel name order, even if the channel_name_to_index is poorly
    sorted. Input checks are also included

    Parameters
    ----------
    channel_name_to_index : dict[str, int]

    Returns
    -------
    cdl_eeg.models.region_based_pooling.utils.CHANNELS_NAMES

    Examples
    --------
    >>> get_channel_name_order(channel_name_to_index={"C2": 2, "C0": 0, "C1": 1, "C4": 3})
    ('C0', 'C1', 'C2', 'C4')

    If there are too small or large values, a ValueError is raised

    >>> get_channel_name_order(channel_name_to_index={"C2": 2, "C0": -2, "C1": 1, "C4": 3})
    Traceback (most recent call last):
    ...
    ValueError: Expected all values to be between 0 and 3, but found (2, -2, 1, 3)
    >>> get_channel_name_order(channel_name_to_index={"C2": 2, "C0": 0, "C1": 1, "C4": 5})
    Traceback (most recent call last):
    ...
    ValueError: Expected all values to be between 0 and 3, but found (2, 0, 1, 5)

    Duplicates are not allowed

    >>> get_channel_name_order(channel_name_to_index={"C2": 1, "C0": 0, "C1": 1, "C4": 3})
    Traceback (most recent call last):
    ...
    ValueError: Expected all values to be unique, but number of channel names was 4 and number of unique values were 3
    """
    # ----------------
    # Input checks
    # ----------------
    num_channels = len(channel_name_to_index)

    # All channel names must be strings
    if not all(ch_name for ch_name in channel_name_to_index):
        raise TypeError(f"Expected all channel names to be strings, but found "
                        f"{set(ch_name for ch_name in channel_name_to_index)}")

    # All indices must be in [0, num_channels - 1]
    if not all(idx in range(num_channels) for idx in channel_name_to_index.values()):
        raise ValueError(f"Expected all values to be between 0 and {num_channels-1}, but found "
                         f"{tuple(channel_name_to_index.values())}")

    # All indices must be integers
    if not all(isinstance(idx, int) for idx in channel_name_to_index.values()):
        raise TypeError(f"Expected all values to be integers, but found "
                        f"{set(type(idx) for idx in channel_name_to_index.values())}")

    # No duplicates of indices
    if len(channel_name_to_index) != len(set(channel_name_to_index.values())):
        raise ValueError(f"Expected all values to be unique, but number of channel names was "
                         f"{len(channel_name_to_index)} and number of unique values were "
                         f"{len(set(channel_name_to_index.values()))}")

    # ----------------
    # Invert the dict
    # ----------------
    inverted_dict = {idx: ch_name for ch_name, idx in channel_name_to_index.items()}

    # Ensure correct ordering by looping through 0 to num_channels - 1
    return tuple(inverted_dict[i] for i in range(num_channels))


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


# ------------------
# Errors
# ------------------
class DataError(Exception):
    ...
