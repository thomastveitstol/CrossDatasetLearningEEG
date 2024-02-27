import os
import random

import mne
import numpy
import pytest
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from cdl_eeg.data.combined_datasets import LoadDetails, CombinedDatasets
from cdl_eeg.data.data_generators.data_generator import SelfSupervisedDataGenerator
from cdl_eeg.data.subject_split import KFoldDataSplit
from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.datasets.van_hees import VanHees
from cdl_eeg.data.paths import get_raw_data_storage_path
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel, tensor_dict_to_device
from cdl_eeg.models.region_based_pooling.region_based_pooling import RBPDesign, RBPPoolType
from cdl_eeg.models.transformations.frequency_slowing import FrequencySlowing
from cdl_eeg.models.transformations.utils import UnivariateUniform


@pytest.mark.xfail
def test_forward():
    """Test forward method when only one channel system has been fit"""
    # ----------------
    # Generate dummy channel system
    # ----------------
    # Channel system 1
    montage_1 = mne.channels.make_standard_montage("GSN-HydroCel-129")
    electrode_positions_1 = montage_1.get_positions()["ch_pos"]
    ch_name_to_idx_1 = {name: i for i, name in enumerate(electrode_positions_1.positions)}
    channel_system_1 = ChannelSystem(name="TestName1", channel_name_to_index=ch_name_to_idx_1,
                                     electrode_positions=electrode_positions_1)

    # Channel system 2
    montage_2 = mne.channels.make_standard_montage("biosemi64")
    electrode_positions_2 = montage_2.get_positions()["ch_pos"]
    ch_name_to_idx_2 = {name: i for i, name in enumerate(electrode_positions_2.positions)}
    channel_system_2 = ChannelSystem(name="TestName2", channel_name_to_index=ch_name_to_idx_2,
                                     electrode_positions=electrode_positions_2)

    # All channel systems
    channel_systems = (channel_system_1, channel_system_2)
    channel_name_to_index = {"TestName1": ch_name_to_idx_1, "TestName2": ch_name_to_idx_2}

    # Data
    time_steps = 2_113
    batch_size_1, batch_size_2 = 31, 13

    data_1 = torch.rand(size=(batch_size_1, len(electrode_positions_1), time_steps))
    data_2 = torch.rand(size=(batch_size_2, len(electrode_positions_1), time_steps))

    data = {"TestName1": data_1, "TestName2": data_2}

    # ----------------
    # Make RBP designs
    # ----------------
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    # Design 1
    num_regions_1 = (5, 5, 5)
    num_channel_splits_1 = len(num_regions_1)

    design_1 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_1, "num_kernels": 43, "max_receptive_field": 37},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_1)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_1)
    )

    # Design 2
    num_regions_2 = 3
    num_designs_2 = 7

    design_2 = RBPDesign(
        pooling_type=RBPPoolType.SINGLE_CS, pooling_methods="SingleCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_2, "num_kernels": 42, "max_receptive_field": 67},
        split_methods="VoronoiSplit",
        split_methods_kwargs={"num_points": num_regions_2, **box_params},
        num_designs=num_designs_2
    )

    # Design 3
    num_regions_3 = (5, 5, 3, 5, 6, 3)
    num_channel_splits_3 = len(num_regions_3)

    design_3 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_3, "num_kernels": 5, "max_receptive_field": 55},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_3)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_3)
    )

    # Design 4
    num_regions_4 = 4
    num_designs_4 = 5

    design_4 = RBPDesign(
        pooling_type=RBPPoolType.SINGLE_CS, pooling_methods="SingleCSMean",
        pooling_methods_kwargs={},
        split_methods="VoronoiSplit",
        split_methods_kwargs={"num_points": num_regions_4, **box_params},
        num_designs=num_designs_4
    )

    # ----------------
    # Make model and fit channel system
    # ----------------
    num_classes = 5
    num_regions = num_regions_1 + (num_regions_2,) * num_designs_2 + num_regions_3 + (num_regions_4,) * num_designs_4

    mts_module = "InceptionNetwork"
    mts_module_kwargs = {"in_channels": sum(num_regions), "num_classes": num_classes}

    model = MainRBPModel(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs,
                         rbp_designs=(design_1, design_2, design_3, design_4))
    model.fit_channel_systems(channel_systems)

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = model.pre_compute(data)
    outputs = model(data, channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, torch.Tensor), (f"Expected the output to be a torch.Tensor, but outputs had type "
                                               f"{type(outputs)}")

    # Shape check
    expected_size = torch.Size([(batch_size_1 + batch_size_2), num_classes])
    assert outputs.size() == expected_size, f"Expected output to have shape {expected_size}, but found {outputs.size()}"


@pytest.mark.xfail
@pytest.mark.skipif(not os.path.isdir(get_raw_data_storage_path()), reason="Required datasets not available")
def test_fit_real_channel_systems():
    """Test fitting of real channel systems"""
    # ------------------
    # Create real channel systems
    # ------------------
    # Select datasets for testing
    datasets = (VanHees(), Rockhill(), Miltiadous())

    # Get their channel systems
    channel_systems = tuple(dataset.channel_system for dataset in datasets)

    # ----------------
    # Make model supporting RBP  TODO: copied
    # ----------------
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    # Design 1
    num_regions_1 = (5, 5, 5)
    num_channel_splits_1 = len(num_regions_1)

    design_1 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_1, "num_kernels": 43, "max_receptive_field": 37},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_1)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_1)
    )

    # Design 2
    num_regions_2 = 3
    num_designs_2 = 7

    design_2 = RBPDesign(
        pooling_type=RBPPoolType.SINGLE_CS, pooling_methods="SingleCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_2, "num_kernels": 93, "max_receptive_field": 67},
        split_methods="VoronoiSplit",
        split_methods_kwargs={"num_points": num_regions_2, **box_params},
        num_designs=num_designs_2
    )

    # Design 3
    num_regions_3 = (5, 5, 3, 5, 6, 3)
    num_channel_splits_3 = len(num_regions_3)

    design_3 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_3, "num_kernels": 5, "max_receptive_field": 55},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_3)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_3)
    )

    # Design 4
    num_regions_4 = 4
    num_designs_4 = 5

    design_4 = RBPDesign(
        pooling_type=RBPPoolType.SINGLE_CS, pooling_methods="SingleCSMean",
        pooling_methods_kwargs={},
        split_methods="VoronoiSplit",
        split_methods_kwargs={"num_points": num_regions_4, **box_params},
        num_designs=num_designs_4
    )

    # ----------------
    # Make model and fit channel system
    # ----------------
    num_classes = 5
    num_regions = num_regions_1 + (num_regions_2,) * num_designs_2 + num_regions_3 + (num_regions_4,) * num_designs_4

    mts_module = "InceptionNetwork"
    mts_module_kwargs = {"in_channels": sum(num_regions), "num_classes": num_classes}

    model = MainRBPModel(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs,
                         rbp_designs=(design_1, design_2, design_3, design_4))
    model.fit_channel_systems(channel_systems)

    # ----------------
    # Tests
    # ----------------
    for channel_system in channel_systems:
        assert all(channel_system.name in rbp_module.channel_splits
                   for rbp_module in model._region_based_pooling._rbp_modules), \
            (f"Expected all channel systems to be fit to all RBP modules, but this was not the case for the channel "
             f"system {channel_system.name}")


@pytest.mark.xfail
@pytest.mark.skipif(not os.path.isdir(get_raw_data_storage_path()), reason="Required datasets not available")
def test_pre_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(2)
    numpy.random.seed(2)

    # ------------------
    # Load data  todo: use more datasets when supported
    # ------------------
    datasets = (Miltiadous(), Rockhill())
    channel_name_to_index = {dataset.name: dataset.channel_system.channel_name_to_index for dataset in datasets}

    # Defining load details
    num_subjects = (25, 14)
    num_time_steps = 1_003
    subjects = {dataset.name: dataset.get_subject_ids()[:subjects] for dataset, subjects in zip(datasets, num_subjects)}

    load_details = tuple(LoadDetails(subject_ids=subject_ids, num_time_steps=num_time_steps)
                         for subject_ids in subjects.values())

    # Load all data
    combined_dataset = CombinedDatasets(datasets, load_details=load_details)

    # Split the data
    data_split = KFoldDataSplit(num_folds=3, dataset_subjects=subjects, seed=2).folds

    # Split into training and validation splits
    train_subjects = data_split[0]
    val_subjects = data_split[1]

    # Define pretext task/transformation
    slowing_distribution = UnivariateUniform(lower=.6, upper=.8)
    transformation = FrequencySlowing(slowing_distribution=slowing_distribution)
    pretext_task = "phase_slowing"

    # ----------------
    # Make model supporting RBP  TODO: works only on RBPPoolType.MULTI_CS
    # ----------------
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    # Design 1
    num_regions_1 = (5, 5, 5)
    num_channel_splits_1 = len(num_regions_1)

    design_1 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_1, "num_kernels": 43, "max_receptive_field": 37},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_1)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_1)
    )

    # Design 2
    num_regions_2 = (5, 5)
    num_channel_splits_2 = len(num_regions_2)

    design_2 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_2, "num_kernels": 53, "max_receptive_field": 27},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_2)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_2)
    )

    # Design 3
    num_regions_3 = 5
    num_designs_3 = 3

    design_3 = RBPDesign(
        pooling_type=RBPPoolType.SINGLE_CS, pooling_methods="SingleCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_3, "num_kernels": 5, "max_receptive_field": 55},
        split_methods="VoronoiSplit",
        split_methods_kwargs={"num_points": num_regions_3, **box_params},
        num_designs=num_designs_3
    )

    # Design 4  todo: the pytorch DataLoader cannot handle None values in pre_computing
    # num_regions_4 = 4
    # num_designs_4 = 5
    #
    # design_4 = RBPDesign(
    #     pooling_type=RBPPoolType.SINGLE_CS, pooling_methods="SingleCSMean",
    #     pooling_methods_kwargs={},
    #     split_methods="VoronoiSplit",
    #     split_methods_kwargs={"num_points": num_regions_4, **box_params},
    #     num_designs=num_designs_4
    # )

    # Some hyperparameters
    num_classes = 1
    num_regions = num_regions_1 + num_regions_2 + (num_regions_3,) * num_designs_3

    mts_module = "InceptionNetwork"
    mts_module_kwargs = {"in_channels": sum(num_regions), "num_classes": num_classes}

    # Make model and fit channel systems
    model = MainRBPModel(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs,
                         rbp_designs=(design_1, design_2, design_3)).to(device)
    model.fit_channel_systems(tuple(dataset.channel_system for dataset in datasets))

    # ----------------
    # Perform pre-training
    # ----------------
    # Extract numpy array data
    train_data = combined_dataset.get_data(subjects=train_subjects)
    val_data = combined_dataset.get_data(subjects=val_subjects)

    # Perform pre-computing
    train_pre_computed = model.pre_compute(input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                                                          for dataset_name, data in train_data.items()})
    val_pre_computed = model.pre_compute(input_tensors={dataset_name: torch.tensor(data, dtype=torch.float).to(device)
                                                        for dataset_name, data in val_data.items()})

    # Send to cpu
    train_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                               for pre_comp in train_pre_computed)
    val_pre_computed = tuple(tensor_dict_to_device(pre_comp, device=torch.device("cpu"))
                             for pre_comp in val_pre_computed)

    # Create data generators
    train_gen = SelfSupervisedDataGenerator(data=train_data, pre_computed=train_pre_computed,
                                            transformation=transformation, pretext_task=pretext_task)
    val_gen = SelfSupervisedDataGenerator(data=val_data, pre_computed=val_pre_computed, transformation=transformation,
                                          pretext_task=pretext_task)

    # Create data loaders
    train_loader = DataLoader(dataset=train_gen, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_gen, batch_size=20, shuffle=True)

    # Create optimiser and loss
    optimiser = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss(reduction="mean")

    # Pre-train
    model.train_model(train_loader=train_loader, val_loader=val_loader, metrics="regression", criterion=criterion,
                      optimiser=optimiser, num_epochs=5, verbose=False, channel_name_to_index=channel_name_to_index,
                      device=device)
