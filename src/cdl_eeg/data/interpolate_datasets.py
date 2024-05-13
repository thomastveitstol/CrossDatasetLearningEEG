"""
Functions and classes for interpolating all datasets to the same channel system
"""
from typing import List

import mne
import numpy

from cdl_eeg.data.datasets.dataset_base import ChannelSystem, get_channel_name_order


def interpolate_datasets(datasets, main_channel_system, method, sampling_freq):
    """
    Function for interpolating all datasets to a single main channel system

    (unit test in test folder)

    Parameters
    ----------
    datasets : dict[str, dict[str, numpy.ndarray | ChannelSystem]]
    main_channel_system : ChannelSystem
    method : str
    sampling_freq : float

    Returns
    -------
    dict[str, numpy.ndarray]
    """
    # -------------
    # Input checks
    # -------------
    # Check that all keys are strings (should be dataset names)
    if not all(isinstance(name, str) for name in datasets):
        raise TypeError(f"Expected all keys to be dataset names of type 'str', but found "
                        f"{set(type(name) for name in datasets)}")

    # Checks per dataset
    for name, dataset in datasets.items():
        # Type check all channel systems
        if not isinstance(dataset["channel_system"], ChannelSystem):
            raise TypeError(f"Expected the dataset to have a channel system of type '{ChannelSystem.__name__}', but "
                            f"found {type(dataset['channel_system'])} for the dataset {name}")

        # Check that all datasets are 4D
        if dataset["data"].ndim != 4:
            raise ValueError(f"Expected data to be 3D, but found {dataset['data'].ndim}D for the dataset {name}")

    # -------------
    # Interpolate all datasets
    # -------------
    return {name: _interpolate_single_dataset(dataset=dataset, to_channel_system=main_channel_system, method=method,
                                              sampling_freq=sampling_freq) for name, dataset in datasets.items()}


def _interpolate_single_dataset(dataset, to_channel_system, method, sampling_freq):
    """
    Function for interpolating a single dataset

    Parameters
    ----------
    dataset : dict[str, numpy.ndarray | ChannelSystem]
    to_channel_system : ChannelSystem
    method : str
    sampling_freq : float

    Returns
    -------
    numpy.ndarray
    """
    eeg_data: numpy.ndarray = dataset["data"]  # type: ignore[type-arg]
    source_channel_system: ChannelSystem = dataset["channel_system"]

    # Input checks
    if source_channel_system.montage_name is None:
        raise ValueError("The channel system must have a valid montage name, but found None")
    if to_channel_system.montage_name is None:
        raise ValueError("The channel system must have a valid montage name, but found None")

    # --------------
    # Create target montage
    # --------------
    # Create info
    target_ch_names = get_channel_name_order(to_channel_system.channel_name_to_index)
    target_info = mne.create_info(ch_names=target_ch_names, sfreq=sampling_freq, ch_types="eeg")

    # Set the montage
    # todo: currently forcing use of MNE default montage
    target_info.set_montage(mne.channels.make_standard_montage(kind=to_channel_system.montage_name), verbose=False)
    target_montage = target_info.get_montage()

    # --------------
    # Create source info
    # --------------
    source_ch_names = get_channel_name_order(source_channel_system.channel_name_to_index)
    source_montage = mne.channels.make_standard_montage(kind=source_channel_system.montage_name)
    source_info = mne.create_info(ch_names=source_ch_names, sfreq=sampling_freq, ch_types="eeg")
    source_info.set_montage(source_montage)

    # --------------
    # Map all EEGs to target montage/channel system
    # --------------
    # Loop through all EEGs in the dataset
    interpolated_data: List[numpy.ndarray] = []  # type: ignore[type-arg]
    for subject_eeg in eeg_data:
        # Create raw
        source_raw = mne.EpochsArray(data=subject_eeg, info=source_info, verbose=False)

        # Perform mapping
        mapped_raw = _mne_map_montage(source_data=source_raw, target_montage=target_montage, method=method)

        # Append interpolated data
        interpolated_data.append(numpy.expand_dims(mapped_raw.get_data(), axis=0))

    # Concatenate to a single ndarray and return
    return numpy.concatenate(interpolated_data, axis=0)


# ---------------
# Functions mainly operating on MNE objects
# ---------------
def _mne_map_montage(source_data, target_montage, method):
    """
    Map EEG from one montage (source) to another (target).

    The steps of this function is as follows:
        1) Create a channel system containing the channels of both the source and the target channel systems. The
            signal values of the channels of the source dataset is kept, while the target channels are zero-filled.
        2) Set the target channels to 'bad', and interpolate them using MNE.
        3) Remove the source channels

    Parameters
    ----------
    source_data : mne.EpochsArray
    target_montage : mne.channels.DigMontage
    method : str

    Returns
    -------
    mne.EpochsArray

    Examples
    --------
    >>> # Create source data
    >>> my_source_channels = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3',
    ...                       'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8',
    ...                       'AF4', 'Fp2', 'Fz', 'Cz']
    >>> my_source_montage = mne.channels.make_standard_montage("standard_1020")
    >>> my_source_info = mne.create_info(ch_names=my_source_channels, sfreq=500, ch_types="eeg")
    >>> _ = my_source_info.set_montage(my_source_montage, verbose=False)
    >>> my_source_data = mne.EpochsArray(data=numpy.random.normal(0, 1, size=(5, len(my_source_channels), 2000)),
    ...                                  info=my_source_info, verbose=False)
    >>> # Create target montage
    >>> my_target_channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz",
    ...                       "C4", "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1",
    ...                       "Oz", "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FC3",
    ...                       "FC4", "FT8", "C5", "C1", "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2",
    ...                       "P6", "PO7", "PO3", "POz", "PO4", "PO8"]
    >>> my_target_info = mne.create_info(ch_names=my_target_channels, sfreq=500, ch_types="eeg")
    >>> _ = my_target_info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    >>> my_target_montage = my_target_info.get_montage()
    >>> # Perform interpolation to target montage
    >>> my_transformed_data = _mne_map_montage(source_data=my_source_data, target_montage=my_target_montage,
    ...                                        method="MNE")
    >>> my_transformed_data.get_data().shape == (5, len(my_target_channels), 2000)
    True
    >>> my_transformed_data.ch_names == my_target_channels
    True
    """
    # Input check
    if source_data.get_montage() is None:
        raise ValueError("The source data must contain a montage, but found None")

    source_montage = source_data.get_montage().copy()

    # -------------
    # Step 1: Create montage containing channels from source and target channel system
    # -------------
    combined_montage = _combine_mne_montages(source_montage=source_montage, target_montage=target_montage)

    # -------------
    # Step 2: Set target channels to bad and interpolate
    # -------------
    # Create info object, set bads and montage
    combined_info = mne.create_info(ch_names=combined_montage.ch_names, sfreq=source_data.info["sfreq"], ch_types="eeg")
    combined_info["bads"] = target_montage.ch_names
    combined_info.set_montage(combined_montage)

    # Create RawArray object and interpolate
    num_target_channels = len(target_montage.ch_names)
    num_time_steps = source_data.get_data().shape[-1]
    num_eeg_epochs = source_data.get_data().shape[0]
    combined_data = numpy.concatenate(
        (source_data.get_data(), numpy.zeros(shape=(num_eeg_epochs, num_target_channels, num_time_steps))), axis=1
    )
    combined_eeg = mne.EpochsArray(data=combined_data, info=combined_info, verbose=False)
    combined_eeg.interpolate_bads(method={"eeg": method}, verbose=False)

    # -------------
    # Step 3: Remove source channels
    # -------------
    # Create info object
    info = mne.create_info(ch_names=target_montage.ch_names, sfreq=source_data.info["sfreq"], ch_types="eeg")

    # Create RawArray object, set montage, and return
    data = combined_eeg.get_data(picks=target_montage.ch_names)
    eeg = mne.EpochsArray(data=data, info=info, verbose=False)
    eeg.set_montage(target_montage, verbose=False)

    return eeg


def _combine_mne_montages(source_montage, target_montage):
    """
    Combine two montages

    todo: must make tests and plots
    Parameters
    ----------
    source_montage : mne.channels.DigMontage
    target_montage : mne.channels.DigMontage

    Returns
    -------
    mne.channels.DigMontage
    """
    # When adding together two montages, MNE does not accept equal channel names. Therefore, renaming the channels of
    # the source montage
    source_montage.rename_channels({ch_name: f"{ch_name}_source" for ch_name in source_montage.ch_names})

    # Add them together
    return source_montage + target_montage
