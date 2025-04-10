import os

import pytest
import torch

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.td_brain import TDBrain
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang

from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.paths import get_raw_data_storage_path
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel


@pytest.mark.skipif(not os.path.isdir(get_raw_data_storage_path()), reason="Required datasets not available")
def test_fit_real_channel_systems(rbp_model_configs):
    """Test fitting of real channel systems"""
    # ------------------
    # Create real channel systems
    # ------------------
    # Select datasets for testing
    datasets = (Miltiadous(), HatlestadHall(), TDBrain(), YulinWang(), MPILemon())

    # Get their channel systems
    channel_systems = tuple(dataset.channel_system for dataset in datasets)

    # ----------------
    # Make model supporting RBP
    # ----------------
    rbp_config, mts_config = rbp_model_configs
    model = MainRBPModel.from_config(rbp_config=rbp_config, mts_config=mts_config, discriminator_config=None)
    model.fit_channel_systems(channel_systems)

    # ----------------
    # Tests
    # ----------------
    for channel_system in channel_systems:
        assert all(channel_system.name in rbp_module.channel_splits
                   for rbp_module in model._region_based_pooling._rbp_modules), \
            (f"Expected all channel systems to be fit to all RBP modules, but this was not the case for the channel "
             f"system {channel_system.name}")


@pytest.mark.skipif(not os.path.isdir(get_raw_data_storage_path()), reason="Required datasets not available")
def test_forward_shape(fitted_rbp_model, all_datasets, dummy_data, rbp_model_configs):
    # Get the channel names to indices
    channel_name_to_index = {dataset.name: dataset.channel_system.channel_name_to_index for dataset in all_datasets}

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = fitted_rbp_model.pre_compute(dummy_data)
    outputs = fitted_rbp_model(dummy_data, channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, torch.Tensor), (f"Expected the output to be a torch.Tensor, but outputs had type "
                                               f"{type(outputs)}")

    # Shape check
    batch_size = sum(data.size()[0] for data in dummy_data.values())
    expected_size = torch.Size((batch_size, rbp_model_configs[1]["kwargs"]["num_classes"]))
    assert outputs.size() == expected_size, f"Expected output to have shape {expected_size}, but found {outputs.size()}"


def test_forward_manipulation(dummy_data_1, dummy_data_2, dummy_fitted_rbp_model, dummy_eeg_dataset_1,
                              dummy_eeg_dataset_2):
    """Test if manipulating the input of an EEG changes the predictions made on that and only that EEG"""
    channel_name_to_index = {dataset.name: dataset.channel_name_to_index()
                             for dataset in (dummy_eeg_dataset_1, dummy_eeg_dataset_2)}
    input_data = {dummy_eeg_dataset_1.name: dummy_data_1,
                  dummy_eeg_dataset_2.name: dummy_data_2}

    assert isinstance(dummy_fitted_rbp_model, MainRBPModel)
    if dummy_fitted_rbp_model.supports_precomputing:
        pre_computed = dummy_fitted_rbp_model.pre_compute(input_data)
    else:
        pre_computed = None

    # --------------
    # Test
    # --------------
    dummy_fitted_rbp_model.eval()
    outputs_1 = dummy_fitted_rbp_model(
        input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
        use_domain_discriminator=False)

    # Make a change to the input data. The keys should ("DummyDataset", "DummyDataset2"), in that order
    new_input_data = {"DummyDataset1": input_data["DummyDataset1"].clone(),
                      "DummyDataset2": input_data["DummyDataset2"].clone()}
    new_input_data["DummyDataset2"][-3] = torch.rand(size=(new_input_data["DummyDataset2"][-3].size()))

    outputs_2 = dummy_fitted_rbp_model(
        new_input_data, pre_computed=pre_computed, channel_name_to_index=channel_name_to_index,
        use_domain_discriminator=False)

    assert not torch.equal(outputs_1[-3], outputs_2[-3]), \
        (f"Model prediction was the same after changing the input of the model {dummy_fitted_rbp_model}"
         f"\n{outputs_1-outputs_2}")
    assert torch.equal(outputs_1[:-3], outputs_2[:-3]), \
        (f"Changing the input of a subject lead to changes for other subjects for model {dummy_fitted_rbp_model}\n"
         f"{outputs_1-outputs_2}")
    assert torch.equal(outputs_1[-2:], outputs_2[-2:]), \
        (f"Changing the input of a subject lead to changes for other subjects for model {dummy_fitted_rbp_model}\n"
         f"{outputs_1-outputs_2}")
