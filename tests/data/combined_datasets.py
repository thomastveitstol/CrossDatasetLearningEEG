import os

import pytest

from cdl_eeg.data.combined_datasets import LoadDetails, CombinedDatasets
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.paths import get_numpy_data_storage_path


@pytest.mark.skipif(not os.path.isdir(get_numpy_data_storage_path()), reason="Required numpy arrays not available")
def test_data_loading():
    """Test that the __init__ method runs"""
    # ------------------
    # Define the datasets to test
    # ------------------
    # Selecting datasets  todo: VanHees fails the test...
    datasets = (Rockhill(), Miltiadous(), HatlestadHall())

    # Defining load details
    details_1 = LoadDetails(subject_ids=Rockhill().get_subject_ids()[:9], num_time_steps=1_003)
    details_2 = LoadDetails(subject_ids=Miltiadous().get_subject_ids()[:14], num_time_steps=305)
    details_3 = LoadDetails(subject_ids=HatlestadHall().get_subject_ids()[:3], time_series_start=1_000,
                            num_time_steps=975)

    # ------------------
    # Load all data
    # ------------------
    combined_dataset = CombinedDatasets(datasets, load_details=(details_1, details_2, details_3))

    # ------------------
    # Tests
    # ------------------
    # Check that all datasets are contained in the loaded numpy arrays dict
    assert all(dataset.name in combined_dataset._data for dataset in datasets)

    # Check shapes of numpy arrays
    subjects_1, _, time_steps_1 = combined_dataset._data[Rockhill().name].shape
    subjects_2, _, time_steps_2 = combined_dataset._data[Miltiadous().name].shape
    subjects_3, _, time_steps_3 = combined_dataset._data[HatlestadHall().name].shape

    assert (subjects_1, time_steps_1) == (9, 1_003), "Wrong shape of numpy array"
    assert (subjects_2, time_steps_2) == (14, 305), "Wrong shape of numpy array"
    assert (subjects_3, time_steps_3) == (3, 975), "Wrong shape of numpy array"
