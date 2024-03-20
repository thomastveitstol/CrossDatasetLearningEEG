import os

import numpy
import pytest

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang
from cdl_eeg.data.interpolate_datasets import interpolate_datasets
from cdl_eeg.data.paths import get_numpy_data_storage_path


@pytest.mark.skipif(not os.path.isdir(get_numpy_data_storage_path()), reason="Required numpy arrays not available")
def test_interpolate_datasets():
    """Test interpolation from datasets to a single target channel system"""
    # ----------------
    # Hyperparameters
    # ----------------
    num_subjects = 4  # Will be per dataset
    method = "MNE"

    datasets = HatlestadHall(), MPILemon(), Rockhill(), YulinWang()
    preprocessed_version = "debug_preprocessed_2024-03-07_134145"

    # ----------------
    # Load data
    # ----------------
    loaded_data = dict()
    for dataset in datasets:
        subjects = dataset.get_subject_ids()[:num_subjects]

        # todo: this must be updated
        channel_system = ChannelSystem(name=dataset.channel_system.name,
                                       channel_name_to_index=dataset.channel_system.channel_name_to_index,
                                       electrode_positions=dataset.channel_system.electrode_positions,
                                       montage_name="standard_1020")
        loaded_data[dataset.name] = {"data": dataset.load_numpy_arrays(subject_ids=subjects,
                                                                       pre_processed_version=preprocessed_version),
                                     "channel_system": channel_system}

    # ----------------
    # Map all to MPI Lemon
    # ----------------
    main_dataset = MPILemon()
    main_channel_system = ChannelSystem(name=main_dataset.channel_system.name,
                                        channel_name_to_index=main_dataset.channel_system.channel_name_to_index,
                                        electrode_positions=main_dataset.channel_system.electrode_positions,
                                        montage_name="standard_1020")

    interpolated_data = interpolate_datasets(
        datasets=loaded_data, main_channel_system=main_channel_system, method=method, sampling_freq=9.5111534412567
    )  # todo: hard-coded sampling frequency

    # ----------------
    # Check outputs
    # ----------------
    # Test keys
    assert set(interpolated_data) == {"rockhill", "yulin_wang", "hatlestad_hall", "mpi_lemon"}

    # Type check of all arrays
    assert all(isinstance(arr, numpy.ndarray) for arr in interpolated_data.values())

    # Check spatial dimension
    num_expected_channels = len(main_dataset.channel_name_to_index())
    assert all(arr.shape[1] == num_expected_channels for arr in interpolated_data.values())

    # Check batch dimension
    assert all(arr.shape[0] == num_subjects for arr in interpolated_data.values())
