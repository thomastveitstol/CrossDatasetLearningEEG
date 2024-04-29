import os

import pytest

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.datasets.dataset_base import get_channel_name_order
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.td_brain import TDBrain
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang
from cdl_eeg.data.paths import get_raw_data_storage_path


@pytest.mark.skipif(not os.path.isdir(get_raw_data_storage_path()), reason="Required datasets not available")
@pytest.mark.skipif("TOX_ENV_NAME" in os.environ, reason="Too time consuming to run this test")
def test_channel_names_ordering():
    """Test if the channel names are always the same per dataset (equal and correctly ordered). MPI Lemon is tested
    separately, as interpolation must be performed"""
    datasets = {HatlestadHall(): {"derivatives": True, "session": "t1"},
                YulinWang(): {"derivatives": True, "visit": 1, "recording": "EC"},
                Miltiadous(): {"derivatives": True},
                CAUEEG(): {"derivatives": False},
                TDBrain(): {"derivatives": False}}

    # Loop through all datasets
    for dataset, kwargs in datasets.items():
        # Get the channel names
        expected_channel_names = get_channel_name_order(dataset.channel_name_to_index())

        # Loop through all subjects
        for subject_id in dataset.get_subject_ids():
            # Load the EEG object
            raw = dataset.load_single_mne_object(subject_id=subject_id, **kwargs, preload=False)

            # Test if the channel names are as expected
            assert tuple(raw.ch_names) == expected_channel_names, \
                (f"The loaded channel names did not match the expected ones for the dataset {dataset.name}:\n"
                 f"Actual: {raw.ch_names}\nExpected: {expected_channel_names}")


@pytest.mark.skipif(not os.path.isdir(get_raw_data_storage_path()), reason="Required datasets not available")
@pytest.mark.skipif("TOX_ENV_NAME" in os.environ, reason="Too time consuming to run this test")
def test_channel_names_ordering_mpi_lemon():
    """Test if the channel names are always the for MPI Lemon (note that this is quite slow)"""
    dataset = MPILemon()

    # Get the channel names
    expected_channel_names = get_channel_name_order(dataset.channel_name_to_index())
    assert expected_channel_names == dataset._channel_names, \
        (f"Two different methods for getting the channel names ordering gave different results:\nMethod 1: "
         f"{expected_channel_names}\nMethod 2: {dataset._channel_names}")

    # Loop through all subjects
    for subject_id in dataset.get_subject_ids():
        # Load the EEG object
        raw = dataset.load_single_mne_object(subject_id=subject_id, interpolation_method="MNE", derivatives=False,
                                             preload=True)  # It relies on .get_data() anyway

        # Test if the channel names are as expected
        assert tuple(raw.ch_names) == expected_channel_names, \
            (f"The loaded channel names did not match the expected ones:\n Actual: {raw.ch_names}\n"
             f"Expected: {expected_channel_names}")
