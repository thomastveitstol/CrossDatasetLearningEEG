import os

import numpy
import pytest

from cdl_eeg.data.combined_datasets import LoadDetails, CombinedDatasets
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.td_brain import TDBrain
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang
from cdl_eeg.data.subject_split import Subject
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.paths import get_numpy_data_storage_path


@pytest.mark.skipif(not os.path.isdir(get_numpy_data_storage_path()), reason="Required numpy arrays not available")
@pytest.mark.parametrize("required_target", [None, "age", "sex"])
def test_data_loading(required_target):
    """Test that the __init__ method runs"""
    # ------------------
    # Define the datasets to test
    # ------------------
    # Selecting datasets
    datasets = (MPILemon(), Miltiadous(), HatlestadHall())

    # Defining load details
    version = "preprocessed_2024-05-13_173548/data_band_pass_8-12_autoreject_True_sampling_multiple_4"
    details_1 = LoadDetails(subject_ids=MPILemon().get_subject_ids()[:9], pre_processed_version=version)
    details_2 = LoadDetails(subject_ids=Miltiadous().get_subject_ids()[:14], pre_processed_version=version)
    details_3 = LoadDetails(subject_ids=HatlestadHall().get_subject_ids()[:3], pre_processed_version=version)

    # ------------------
    # Load all data
    # ------------------
    combined_dataset = CombinedDatasets(datasets, load_details=(details_1, details_2, details_3),
                                        required_target=required_target)

    # ------------------
    # Tests
    # ------------------
    # Check that all datasets are contained in the loaded numpy arrays dict
    assert all(dataset.name in combined_dataset._data for dataset in datasets)

    # Check shapes of numpy arrays
    subjects_1, *_, time_steps_1 = combined_dataset._data[MPILemon().name].shape
    subjects_2, *_, time_steps_2 = combined_dataset._data[Miltiadous().name].shape
    subjects_3, *_, time_steps_3 = combined_dataset._data[HatlestadHall().name].shape

    assert subjects_1 == 9, "Wrong number of subjects"
    assert subjects_2 == 14, "Wrong number of subjects"
    assert subjects_3 == 3, "Wrong number of subjects"

    assert time_steps_1 == time_steps_2 == time_steps_3, "Unequal number of time steps"


@pytest.mark.skipif(not os.path.isdir(get_numpy_data_storage_path()), reason="Required numpy arrays not available")
@pytest.mark.parametrize("required_target", [None, "age", "sex"])
def test_get_data(required_target):
    """Tests for the get_data method"""
    # ------------------
    # Define the datasets to test
    # ------------------
    # Selecting datasets
    datasets = (MPILemon(), Miltiadous(), HatlestadHall())

    # Defining load details
    version = "preprocessed_2024-05-13_173548/data_band_pass_8-12_autoreject_True_sampling_multiple_4"
    details_1 = LoadDetails(subject_ids=MPILemon().get_subject_ids()[:9], pre_processed_version=version)
    details_2 = LoadDetails(subject_ids=Miltiadous().get_subject_ids()[:14], pre_processed_version=version)
    details_3 = LoadDetails(subject_ids=HatlestadHall().get_subject_ids()[:3], pre_processed_version=version)

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
    subjects_1, *_, time_steps_1 = extracted_data[Miltiadous().name].shape
    subjects_2, *_, time_steps_2 = extracted_data[HatlestadHall().name].shape

    assert subjects_1 == 10, "Wrong number of subjects"
    assert subjects_2 == 3, "Wrong number of subjects"

    assert time_steps_1 == time_steps_2, "Unequal number of time steps"


@pytest.mark.skipif(not os.path.isdir(get_numpy_data_storage_path()), reason="Required numpy arrays not available")
@pytest.mark.parametrize("target", ["age", "sex"])
def test_get_targets(target):
    """Test if the 'get_targets' method works as expected. Some subjects were manually selected and tested per dataset
    used in the paper"""
    # ------------------
    # Define the datasets to test
    # ------------------
    # Selecting datasets
    datasets = (MPILemon(), Miltiadous(), HatlestadHall(), YulinWang(), TDBrain())

    # Manually select subjects for testing
    subjects = {"MPILemon": ("sub-032301", "sub-032302", "sub-032303", "sub-032304", "sub-032305", "sub-032306",
                             "sub-032307", "sub-032308", "sub-032310"),
                "Miltiadous": ("sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008",
                               "sub-009", "sub-010", "sub-011", "sub-012", "sub-013", "sub-014"),
                "HatlestadHall": ("sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007",
                                  "sub-008", "sub-009", "sub-010", "sub-011", "sub-012"),
                "YulinWang": ("sub-01", "sub-02", "sub-03", "sub-04", "sub-15", "sub-20", "sub-19",
                              "sub-21"),
                "TDBrain": ("sub-19681349", "sub-19681385", "sub-19684666", "sub-19686324", "sub-19701225")}

    # Defining load details
    version = "preprocessed_2024-05-13_173548/data_band_pass_8-12_autoreject_True_sampling_multiple_4"
    details_1 = LoadDetails(subject_ids=subjects["MPILemon"], pre_processed_version=version)
    details_2 = LoadDetails(subject_ids=subjects["Miltiadous"], pre_processed_version=version)
    details_3 = LoadDetails(subject_ids=subjects["HatlestadHall"], pre_processed_version=version)
    details_4 = LoadDetails(subject_ids=subjects["YulinWang"], pre_processed_version=version)
    details_5 = LoadDetails(subject_ids=subjects["TDBrain"], pre_processed_version=version)

    # ------------------
    # Load all data
    # ------------------
    combined_dataset = CombinedDatasets(datasets, load_details=(details_1, details_2, details_3, details_4, details_5),
                                        required_target=target, target=target)

    # ------------------
    # Extract data
    # ------------------
    # Decide which data to extract
    data = {dataset.name: subjects[dataset.name] for dataset in datasets}

    # Convert to correct format
    subjects = []
    for dataset_name, subject_ids in data.items():
        for sub_id in subject_ids:
            subjects.append(Subject(dataset_name=dataset_name, subject_id=sub_id))

    # Perform data extraction
    extracted_targets = combined_dataset.get_targets(tuple(subjects))

    # ------------------
    # Tests
    # ------------------
    # Test if the dataset (keys) are as expected
    assert ({MPILemon().name, Miltiadous().name, HatlestadHall().name, YulinWang().name, TDBrain().name} ==
            set(extracted_targets.keys()))

    # Test the actual data (the TDBrain sexes are 'flipped' due to differences in integer to sex mapping)
    expected_sex = {"MPILemon": numpy.array([1, 1, 1, 0, 1, 1, 0, 1, 1]),
                    "Miltiadous": numpy.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]),
                    "HatlestadHall": numpy.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
                    "YulinWang": numpy.array([0, 1, 1, 1, 0, 0, 1, 0]),
                    "TDBrain": numpy.array([1, 0, 0, 0, 0])}
    expected_age = {"MPILemon": numpy.array([67.5, 22.5, 67.5, 27.5, 67.5, 67.5, 27.5, 62.5, 27.5]),
                    "Miltiadous": numpy.array([57, 78, 70, 67, 70, 61, 79, 62, 77, 69, 71, 63, 64, 77]),
                    "HatlestadHall": numpy.array([29, 29, 62, 20, 32, 39, 37, 34, 19, 34, 46, 32]),
                    "YulinWang": numpy.array([19, 19, 19, 20, 22, 19, 19, 22]),
                    "TDBrain": numpy.array([51.59, 49.96, 47.05, 62.51, 9.0])}
    for dataset, targets in extracted_targets.items():
        if target == "sex":
            assert numpy.array_equal(targets, expected_sex[dataset]), \
                (f"The extracted targets were not as expected for the dataset '{dataset}'. Expected: "
                 f"{expected_sex[dataset]}\nReceived: {targets}")
        elif target == "age":
            assert numpy.array_equal(targets, expected_age[dataset]), \
                (f"The extracted targets were not as expected for the dataset '{dataset}'. Expected: "
                 f"{expected_age[dataset]}\nReceived: {targets}")
        else:
            raise ValueError("Should not happen")
