import numpy.random
import pytest
import torch

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.td_brain import TDBrain
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel


@pytest.fixture
def rbp_model_configs():
    """Configurations for building a MainRBPModel from config"""
    # RBP config
    rbp_config = {
        "RBPDesigns": {
            "RBPDesign0": {
                "cmmn_kwargs": {
                    "kernel_size": 128
                },
                "num_designs": 1,
                "pooling_methods": "MultiMSMean",
                "pooling_methods_kwargs": {},
                "pooling_type": "multi_cs",
                "split_methods": ["CentroidPolygons"],
                "split_methods_kwargs": [{
                    "channel_positions": ["Miltiadous"],
                    "k": [2, 2, 2, 2, 2, 2, 2, 2],
                    "min_nodes": 1
                }],
                "use_cmmn_layer": True
            },
            "RBPDesign1": {
                "cmmn_kwargs": {
                    "kernel_size": 512
                },
                "num_designs": 1,
                "pooling_methods": "MultiMSSharedRocketHeadRegion",
                "pooling_methods_kwargs": {
                    "bias": False,
                    "latent_search_features": 12,
                    "max_receptive_field": 123,
                    "num_kernels": 203,
                    "share_search_receiver_modules": [True, False]
                },
                "pooling_type": "multi_cs",
                "split_methods": ["CentroidPolygons"],
                "split_methods_kwargs": [{
                    "channel_positions": ["Miltiadous"],
                    "k": [2, 2, 2, 2, 2, 2, 2, 2],
                    "min_nodes": 2
                }],
                "use_cmmn_layer": True
            },
            "RBPDesign2": {
                "cmmn_kwargs": {
                    "kernel_size": 128
                },
                "num_designs": 1,
                "pooling_methods": "MultiMSMean",
                "pooling_methods_kwargs": {},
                "pooling_type": "multi_cs",
                "split_methods": ["CentroidPolygons"],
                "split_methods_kwargs": [{
                    "channel_positions": ["Miltiadous"],
                    "k": [2, 3, 2, 3, 2, 3, 2, 3, 2],
                    "min_nodes": 2
                }],
                "use_cmmn_layer": True
            }
        },
        "normalise_region_representations": True}

    # MTS module config
    mts_config = {"model": "InceptionNetwork", "kwargs": {"depth": 12, "cnn_units": 31, "num_classes": 1}}

    return rbp_config, mts_config


@pytest.fixture
def all_datasets():
    return Miltiadous(), HatlestadHall(), TDBrain(), YulinWang(), MPILemon()


@pytest.fixture
def fitted_rbp_model(rbp_model_configs, all_datasets, dummy_data):
    # ------------------
    # Create real channel systems
    # ------------------
    # Get their channel systems
    channel_systems = {dataset.name: dataset.channel_system for dataset in all_datasets}

    # ----------------
    # Make model supporting RBP
    # ----------------
    # Define model
    rbp_config, mts_config = rbp_model_configs
    model = MainRBPModel.from_config(rbp_config=rbp_config, mts_config=mts_config, discriminator_config=None)

    # Fit channel systems
    model.fit_channel_systems(tuple(channel_systems.values()))

    # Fit CMMN layer
    if model.any_rbp_cmmn_layers:
        model.fit_psd_barycenters(data=dummy_data, channel_systems=channel_systems, sampling_freq=128)
        model.fit_monge_filters(data=dummy_data, channel_systems=channel_systems)

    # Return it
    return model


@pytest.fixture
def dummy_data(all_datasets):
    # Some configurations
    num_time_steps = 500
    batch_sizes = tuple(numpy.random.randint(10, 31) for _ in all_datasets)

    # Create random tensors
    data = {}
    for batch_size, dataset in zip(batch_sizes, all_datasets):
        data[dataset.name] = torch.rand(size=(batch_size, dataset.num_channels, num_time_steps))

    return data
