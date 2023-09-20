import mne
import torch

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.main_models.main_rbp_model import MainSingleChannelSplitRBPModel
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
    pooling_methods = ("SingleCSMean", "SingleCSMean", "SingleCSMean", "SingleCSMean")
    pooling_methods_kwargs = ({}, {}, {}, {})
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
    # Run forward method
    # ----------------
    outputs = model(data, channel_system_name=channel_system_name, channel_name_to_index=channel_name_to_index)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, torch.Tensor), (f"Expected the output to be a torch.Tensor, but outputs had type "
                                               f"{type(outputs)}")

    # Shape check
    expected_size = torch.Size([batch_size, num_classes])
    assert outputs.size() == expected_size, f"Expected output to have shape {expected_size}, but found {outputs.size()}"
