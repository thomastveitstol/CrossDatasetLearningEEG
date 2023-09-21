import mne
import torch

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel
from cdl_eeg.models.region_based_pooling.region_based_pooling import RBPDesign, RBPPoolType
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
    # Make RBP designs
    # ----------------
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    # Design 1
    num_regions_1 = (11, 7, 26, 4, 7)
    num_channel_splits_1 = len(num_regions_1)

    design_1 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_1, "num_kernels": 43, "max_receptive_field": 37},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_1)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_1)
    )

    # Design 2
    num_regions_2 = (9, 14)

    design_2 = RBPDesign(
        pooling_type=RBPPoolType.SINGLE_CS, pooling_methods=("SingleCSMean", "SingleCSSharedRocket"),
        pooling_methods_kwargs=({}, {"num_regions": num_regions_2[1], "num_kernels": 93, "max_receptive_field": 67}),
        split_methods=("VoronoiSplit", "VoronoiSplit"),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_2)
    )

    # Design 3
    num_regions_3 = (5, 5, 7, 5, 12, 10)
    num_channel_splits_3 = len(num_regions_3)

    design_3 = RBPDesign(
        pooling_type=RBPPoolType.MULTI_CS, pooling_methods="MultiCSSharedRocket",
        pooling_methods_kwargs={"num_regions": num_regions_3, "num_kernels": 5, "max_receptive_field": 55},
        split_methods=tuple("VoronoiSplit" for _ in range(num_channel_splits_3)),
        split_methods_kwargs=tuple({"num_points": num_regs, **box_params} for num_regs in num_regions_3)
    )

    # ----------------
    # Make model and fit channel system
    # ----------------
    num_classes = 5
    num_regions = num_regions_1 + num_regions_2 + num_regions_3

    mts_module = "InceptionTime"
    mts_module_kwargs = {"in_channels": sum(num_regions), "num_classes": num_classes}

    model = MainRBPModel(mts_module=mts_module, mts_module_kwargs=mts_module_kwargs,
                         rbp_designs=(design_1, design_2, design_3))
    model.fit_channel_system(channel_system)

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = model.pre_compute(data)
    outputs = model(data, channel_system_name=channel_system.name,
                    channel_name_to_index=channel_system.channel_name_to_index, pre_computed=pre_computed)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, torch.Tensor), (f"Expected the output to be a torch.Tensor, but outputs had type "
                                               f"{type(outputs)}")

    # Shape check
    expected_size = torch.Size([batch_size, num_classes])
    assert outputs.size() == expected_size, f"Expected output to have shape {expected_size}, but found {outputs.size()}"
