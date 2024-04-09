import os

import numpy
import pytest

from cdl_eeg.data.combined_datasets import LoadDetails, CombinedDatasets
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.subject_split import Subject
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.paths import get_numpy_data_storage_path


@pytest.mark.skipif(not os.path.isdir(get_numpy_data_storage_path()), reason="Required numpy arrays not available")
def test_data_loading():
    """Test that the __init__ method runs"""
    # ------------------
    # Define the datasets to test
    # ------------------
    # Selecting datasets
    datasets = (Miltiadous(), HatlestadHall())

    # Defining load details
    details_1 = LoadDetails(subject_ids=MPILemon().get_subject_ids()[:9], num_time_steps=1_003)
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
    subjects_1, _, time_steps_1 = combined_dataset._data[MPILemon().name].shape
    subjects_2, _, time_steps_2 = combined_dataset._data[Miltiadous().name].shape
    subjects_3, _, time_steps_3 = combined_dataset._data[HatlestadHall().name].shape

    assert (subjects_1, time_steps_1) == (9, 1_003), "Wrong shape of numpy array"
    assert (subjects_2, time_steps_2) == (14, 305), "Wrong shape of numpy array"
    assert (subjects_3, time_steps_3) == (3, 975), "Wrong shape of numpy array"


@pytest.mark.skipif(not os.path.isdir(get_numpy_data_storage_path()), reason="Required numpy arrays not available")
def test_get_data():
    """Tests for the get_data method"""
    # ------------------
    # Define the datasets to test
    # ------------------
    # Selecting datasets
    datasets = (MPILemon(), Miltiadous(), HatlestadHall())

    # Defining load details
    details_1 = LoadDetails(subject_ids=MPILemon().get_subject_ids()[:9], num_time_steps=1_003)
    details_2 = LoadDetails(subject_ids=Miltiadous().get_subject_ids()[:14], num_time_steps=305)
    details_3 = LoadDetails(subject_ids=HatlestadHall().get_subject_ids()[:3], time_series_start=1_000,
                            num_time_steps=975)

    # ------------------
    # Load all data
    # ------------------
    combined_dataset = CombinedDatasets(datasets, load_details=(details_1, details_2, details_3))

    # ------------------
    # Extract data
    # ------------------
    # Decide which data to extract
    data = {Miltiadous().name: Miltiadous().get_subject_ids()[:10],
            HatlestadHall().name: HatlestadHall().get_subject_ids()[:3]}

    # Convert to correct format
    subjects = []
    for dataset_name, subject_ids in data.items():
        for sub_id in subject_ids:
            subjects.append(Subject(dataset_name=dataset_name, subject_id=sub_id))

    # Perform data extraction
    extracted_data = combined_dataset.get_data(tuple(subjects))

    # ------------------
    # Tests
    # ------------------
    # Test if the datasets asked for are actually in the extracted data
    assert Miltiadous().name in extracted_data and HatlestadHall().name in extracted_data, \
        "Expected dataset(s) not found"

    # Test if the dataset not asked for are not in the extracted data
    assert MPILemon().name not in extracted_data, "Unexpected dataset"

    # Type check of all values
    assert all(isinstance(data_matrix, numpy.ndarray) for data_matrix in extracted_data.values()), \
        "Wrong type (expected numpy.ndarray)"

    # Shape check of values
    subjects_1, _, time_steps_1 = extracted_data[Miltiadous().name].shape
    subjects_2, _, time_steps_2 = extracted_data[HatlestadHall().name].shape

    assert (subjects_1, time_steps_1) == (10, 305), "Wrong shape"
    assert (subjects_2, time_steps_2) == (3, 975), "Wrong shape"
