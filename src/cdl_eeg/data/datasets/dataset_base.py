import abc
import copy
import dataclasses
import os
from typing import Dict, Tuple, List

import enlighten
import inflection
import numpy
import pandas
from matplotlib import pyplot
from mne.transforms import _cart_to_sph, _pol_to_cart

from cdl_eeg.data.paths import get_raw_data_storage_path, get_numpy_data_storage_path
from cdl_eeg.models.region_based_pooling.utils import Electrodes3D


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
    def pre_process(eeg_data, *, remove_above_std, interpolation=None, filtering=None, resample=None, notch_filter=None,
                    avg_reference=False, excluded_channels=None):
        """
        Method for pre-processing EEG data

        Parameters
        ----------
        eeg_data : mne.io.base.BaseRaw
        remove_above_std : float, optional
            Remove channels with standard deviation above this number, and interpolate with the interpolation method
        interpolation : str, optional
            interpolation method to use, if channels are removed
        filtering : tuple[float, float], optional
        resample : float, optional
        notch_filter : float, optional
        avg_reference : bool
        excluded_channels : tuple[str, ...], optional

        Returns
        -------
        mne.io.base.BaseRaw
            The pre-processed MNE raw object
        """
        # TODO: Such shared pre processing steps is not optimal. The EEG data may e.g. contain boundary events or have
        #   unequal requirements such as line noise
        # todo: Maybe try out AutoReject and use spherical spline interpolation?
        # Excluding channels
        if excluded_channels is not None:
            eeg_data = eeg_data.pick(picks="eeg", exclude=excluded_channels)

        if remove_above_std is not None:
            # If there are any currently labelled bad channels, keep them
            bad_channels = set(copy.deepcopy(eeg_data.info["bads"]))

            # Loop through all channels and store the ones which are bad
            for channel in eeg_data.info["ch_names"]:
                channel_data = copy.deepcopy(eeg_data).pick(channel).get_data()[0]
                if numpy.std(channel_data) > remove_above_std:
                    bad_channels.add(channel)

            # Interpolate
            if interpolation is None:
                raise ValueError("Expected an interpolation method, but none was received")
            if bad_channels:
                eeg_data.interpolate_bads(method={"eeg": interpolation})

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

    def load_numpy_arrays(self, subject_ids=None, pre_processed_version=None, *, time_series_start=None,
                          num_time_steps=None, channels=None):
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

        Returns
        -------
        numpy.ndarray
        """
        # Maybe set defaults
        path = self.get_numpy_arrays_path(pre_processed_version)
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

    def save_eeg_as_numpy_arrays(self, path=None, subject_ids=None, *, remove_above_std, filtering=None, resample=None,
                                 notch_filter=None, avg_reference=False, num_time_steps=None, time_series_start=None,
                                 derivatives=False, excluded_channels=None, interpolation=None, **kwargs):
        """
        Method for saving data as numpy arrays

        todo: consider not using so many default arguments

        Parameters
        ----------
        path : str, optional
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
        excluded_channels : tuple[str, ...], optional
            Channels to exclude. If None is passed, no channels will be excluded
        remove_above_std : float, optional
            See preprocessing
        interpolation : str, optional
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
        available_subjects = self.get_subject_ids()
        if not all(sub_id in available_subjects for sub_id in subject_ids):
            _unexpected_subjects = tuple(sub_id for sub_id in self.get_subject_ids() if sub_id not in subject_ids)
            raise ValueError(f"Unexpected subject IDs (N={len(_unexpected_subjects)}): {_unexpected_subjects}")

        # ------------------
        # Prepare directory
        # ------------------
        # Make directory
        path = self.get_numpy_arrays_path() if path is None else os.path.join(path, self.name)
        os.mkdir(path)

        # ------------------
        # Loop through all subjects
        # ------------------
        for sub_id in subject_ids:
            # Load the EEG data as MNE object
            raw = self.load_single_mne_object(subject_id=sub_id, derivatives=derivatives, **kwargs)

            # Pre-process
            raw = self.pre_process(raw, filtering=filtering, resample=resample, notch_filter=notch_filter,
                                   excluded_channels=excluded_channels, avg_reference=avg_reference,
                                   remove_above_std=remove_above_std, interpolation=interpolation)

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

    def get_subject_ids(self) -> Tuple[str, ...]:
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
