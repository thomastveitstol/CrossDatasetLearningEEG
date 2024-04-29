import os

import pytest

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
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
    datasets = (Miltiadous(), CAUEEG(), HatlestadHall(), TDBrain(), YulinWang(), MPILemon())

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
