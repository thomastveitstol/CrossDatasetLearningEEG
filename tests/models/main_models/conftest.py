import mne
import numpy.random
import pytest
import torch

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method
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


# ---------------
# Dummy datasets
# ---------------
@pytest.fixture
def dummy_data_1():
    # Some configurations
    batch_size, num_channels, num_time_steps = 10, 19, 2000

    # Make a random tensor
    return torch.rand(size=(batch_size, num_channels, num_time_steps))


@pytest.fixture
def dummy_data_2():
    # Some configurations
    batch_size, num_channels, num_time_steps = 6, 32, 2000

    # Make a random tensor
    return torch.rand(size=(batch_size, num_channels, num_time_steps))


@pytest.fixture
def dummy_eeg_dataset_1(dummy_data_1):
    _, dummy_num_channels, dummy_num_time_steps = dummy_data_1.size()

    class DummyDataset1(EEGDatasetBase):
        _num_time_steps = dummy_num_time_steps
        _channel_names = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5",
                          "T6", "Fz", "Cz", "Pz")
        _montage_name = "standard_1020"

        # -------------
        # Overriding abstract methods which are not required for these tests
        # -------------
        def _load_single_raw_mne_object(self, *args, **kwargs):
            raise NotImplementedError

        # -------------
        # Overriding methods to make this class suited for testing
        # -------------
        @classmethod
        def _get_template_electrode_positions(cls):
            # Following the standard 10-20 system according to the original article
            montage = mne.channels.make_standard_montage(cls._montage_name)
            channel_positions = montage.get_positions()["ch_pos"]

            # Return dict with channel positions, keeping only the ones in the data
            return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in cls._channel_names}

        def channel_name_to_index(self):
            return {ch_name: i for i, ch_name in enumerate(self._channel_names)}

        def load_numpy_arrays(self, subject_ids=None, pre_processed_version=None, *, time_series_start=None,
                              num_time_steps=None, channels=None, required_target=None):
            return numpy.random.normal(loc=0, scale=1.,
                                       size=(len(subject_ids), len(self._channel_names), self._num_time_steps))

        @target_method
        def age(self, subject_ids):
            return numpy.random.randint(18, 90, size=(len(subject_ids),))

        @target_method
        def sex(self, subject_ids):
            return numpy.random.randint(0, 2, size=(len(subject_ids),))  # 0s and 1s

    return DummyDataset1()



@pytest.fixture
def dummy_eeg_dataset_2(dummy_data_2):
    _, _, dummy_num_time_steps = dummy_data_2.shape

    class DummyDataset2(EEGDatasetBase):
        _num_time_steps = dummy_num_time_steps
        _channel_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz",
                          "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "O2",
                          "FT9", "FT10", "TP9", "TP10")
        _montage_name = "standard_1020"

        # -------------
        # Overriding abstract methods which are not required for these tests
        # -------------
        def _load_single_raw_mne_object(self, *args, **kwargs):
            raise NotImplementedError

        # -------------
        # Overriding methods to make this class suited for testing
        # -------------
        @classmethod
        def _get_template_electrode_positions(cls):
            # Following the standard 10-20 system according to the original article
            montage = mne.channels.make_standard_montage(cls._montage_name)
            channel_positions = montage.get_positions()["ch_pos"]

            # Return dict with channel positions, keeping only the ones in the data
            return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in cls._channel_names}

        def channel_name_to_index(self):
            return {ch_name: i for i, ch_name in enumerate(self._channel_names)}

        def load_numpy_arrays(self, subject_ids=None, pre_processed_version=None, *, time_series_start=None,
                              num_time_steps=None, channels=None, required_target=None):
            return numpy.random.normal(loc=0, scale=1.,
                                       size=(len(subject_ids), len(self._channel_names), self._num_time_steps))

        @target_method
        def age(self, subject_ids):
            return numpy.random.randint(50, 70, size=(len(subject_ids),))

        @target_method
        def sex(self, subject_ids):
            return numpy.random.randint(0, 2, size=(len(subject_ids),))  # 0s and 1s

    return DummyDataset2()


@pytest.fixture
def dummy_fitted_rbp_model(rbp_model_configs, dummy_eeg_dataset_1, dummy_eeg_dataset_2, dummy_data_1, dummy_data_2):
    dummy_data = {dummy_eeg_dataset_1.name: dummy_data_1,
                  dummy_eeg_dataset_2.name: dummy_data_2}
    # ------------------
    # Create real channel systems
    # ------------------
    # Get their channel systems
    channel_systems = {dataset.name: dataset.channel_system for dataset in (dummy_eeg_dataset_1, dummy_eeg_dataset_2)}

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
