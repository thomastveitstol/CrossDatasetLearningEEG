import itertools

import mne
import torch

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.main_models.main_rbp_model import MainSingleChannelSplitRBPModel, MainMultiChannelSplitsRBPModel
from cdl_eeg.models.region_based_pooling.utils import Electrodes3D


def test_forward_single_channel_system():
    """Test forward method when only one channel system has been fit"""
    # ----------------
    # Generate dummy channel system and data
    # ----------------
    # Hyperparameters
    batch_size, time_steps = 10, 2_000
    channel_system_name = "TestName"
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    # Channel system requirements
    electrode_positions = Electrodes3D(montage.get_positions()["ch_pos"])
    channel_name_to_index = {name: i for i, name in enumerate(electrode_positions.positions)}
    channel_system = ChannelSystem(name=channel_system_name, channel_name_to_index=channel_name_to_index,
                                   electrode_positions=electrode_positions)

    # Data
    data = torch.rand(size=(batch_size, len(electrode_positions), time_steps))

    # ----------------
    # Make model and fit channel system
    # ----------------
    # Hyperparameters
    num_regions = 11, 7, 26, 18
    num_classes = 5

    mts_module = "InceptionTime"
    mts_module_kwargs = {"in_channels": sum(num_regions), "num_classes": num_classes}
    pooling_methods = ("SingleCSSharedRocket", "SingleCSMean", "SingleCSMean", "SingleCSSharedRocket")
    pooling_methods_kwargs = ({"num_regions": num_regions[0], "num_kernels": 300, "max_receptive_field": 73},
                              {},
                              {},
                              {"num_regions": num_regions[3], "num_kernels": 23, "max_receptive_field": 131})
    split_methods = ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit", "VoronoiSplit")
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    split_methods_kwargs = ({"num_points": num_regions[0], **box_params}, {"num_points": num_regions[1], **box_params},
                            {"num_points": num_regions[2], **box_params}, {"num_points": num_regions[3], **box_params})

    # Make model
    model = MainSingleChannelSplitRBPModel(
        mts_module=mts_module, mts_module_kwargs=mts_module_kwargs, pooling_methods=pooling_methods,
        pooling_methods_kwargs=pooling_methods_kwargs, split_methods=split_methods,
        split_methods_kwargs=split_methods_kwargs)

    # Fit channel system
    model.fit_channel_system(channel_system)

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = model.pre_compute(data)
    outputs = model(data, channel_system_name=channel_system_name, channel_name_to_index=channel_name_to_index,
                    pre_computed=pre_computed)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, torch.Tensor), (f"Expected the output to be a torch.Tensor, but outputs had type "
                                               f"{type(outputs)}")

    # Shape check
    expected_size = torch.Size([batch_size, num_classes])
    assert outputs.size() == expected_size, f"Expected output to have shape {expected_size}, but found {outputs.size()}"


def test_forward_multi_channel_system():
    """Test forward method when only one channel system has been fit"""
    # ----------------
    # Generate dummy channel system and data
    # ----------------
    # Hyperparameters
    batch_size, time_steps = 10, 2_000
    channel_system_name = "TestName"
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    # Channel system requirements
    electrode_positions = Electrodes3D(montage.get_positions()["ch_pos"])
    channel_name_to_index = {name: i for i, name in enumerate(electrode_positions.positions)}
    channel_system = ChannelSystem(name=channel_system_name, channel_name_to_index=channel_name_to_index,
                                   electrode_positions=electrode_positions)

    # Data
    data = torch.rand(size=(batch_size, len(electrode_positions), time_steps))

    # ----------------
    # Make model and fit channel system
    # ----------------
    # Hyperparameters
    num_regions = ((11, 7, 26), (4, 7), (3, 4, 3, 6))
    num_channel_splits = tuple(len(regions) for regions in num_regions)
    num_classes = 5

    mts_module = "InceptionTime"
    mts_module_kwargs = {"in_channels": sum(tuple(itertools.chain(*num_regions))), "num_classes": num_classes}
    pooling_methods = ("MultiCSSharedRocket", "MultiCSSharedRocket", "MultiCSSharedRocket")
    pooling_methods_kwargs = ({"num_regions": num_regions[0], "num_kernels": 43, "max_receptive_field": 37},
                              {"num_regions": num_regions[1], "num_kernels": 5, "max_receptive_field": 75},
                              {"num_regions": num_regions[2], "num_kernels": 17, "max_receptive_field": 57})
    split_methods = tuple(tuple("VoronoiSplit" for _ in range(num_regs)) for num_regs in num_channel_splits)
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    split_methods_kwargs = tuple(tuple({"num_points": num_regs, **box_params} for num_regs in regions)
                                 for regions in num_regions)

    # Make model
    model = MainMultiChannelSplitsRBPModel(
        mts_module=mts_module, mts_module_kwargs=mts_module_kwargs, pooling_methods=pooling_methods,
        pooling_methods_kwargs=pooling_methods_kwargs, split_methods=split_methods,
        split_methods_kwargs=split_methods_kwargs)

    # Fit channel system
    model.fit_channel_system(channel_system)

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = model.pre_compute(data)
    outputs = model(data, channel_system_name=channel_system_name, channel_name_to_index=channel_name_to_index,
                    pre_computed=pre_computed)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, torch.Tensor), (f"Expected the output to be a torch.Tensor, but outputs had type "
                                               f"{type(outputs)}")

    # Shape check
    expected_size = torch.Size([batch_size, num_classes])
    assert outputs.size() == expected_size, f"Expected output to have shape {expected_size}, but found {outputs.size()}"
