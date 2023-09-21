import mne
import torch

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.region_based_pooling.region_based_pooling import SingleChannelSplitRegionBasedPooling, \
    MultiChannelSplitsRegionBasedPooling, RBPDesign, RBPPoolType, RegionBasedPooling
from cdl_eeg.models.region_based_pooling.utils import Electrodes3D, ChannelsInRegionSplit


# ------------------------
# Testing SingleChannelSplitRegionBasedPooling
# ------------------------
def test_single_cs_fit_channel_system():
    """Tests if a channel system is properly fit. Testing channel_split property types, lengths, that the electrodes
    are contained in regions when expected, and that the number of regions are correct"""
    # ----------------
    # Generate dummy channel system
    # todo: check mCoding if this can be improved
    # ----------------
    channel_system_name = "TestName"
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    electrode_positions = Electrodes3D(montage.get_positions()["ch_pos"])
    channel_name_to_index = {name: i for i, name in enumerate(electrode_positions.positions)}
    channel_system = ChannelSystem(name=channel_system_name, channel_name_to_index=channel_name_to_index,
                                   electrode_positions=electrode_positions)

    # ----------------
    # Make RBP object
    # todo: check mCoding if this can be improved
    # ----------------
    num_regions = 11, 7, 26, 18

    pooling_methods = ("SingleCSMean", "SingleCSMean", "SingleCSMean", "SingleCSMean")
    pooling_methods_kwargs = ({}, {}, {}, {})
    split_methods = ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit", "VoronoiSplit")
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    split_methods_kwargs = ({"num_points": num_regions[0], **box_params}, {"num_points": num_regions[1], **box_params},
                            {"num_points": num_regions[2], **box_params}, {"num_points": num_regions[3], **box_params})

    rbp_module = SingleChannelSplitRegionBasedPooling(pooling_methods=pooling_methods,
                                                      pooling_methods_kwargs=pooling_methods_kwargs,
                                                      split_methods=split_methods,
                                                      split_methods_kwargs=split_methods_kwargs)

    # ----------------
    # Fit channel system
    # ----------------
    rbp_module.fit_channel_system(channel_system=channel_system)

    # ----------------
    # Tests
    # ----------------
    # Check if the channel system is a key in the registered channel splits
    assert channel_system_name in rbp_module.channel_splits, ("Expected the channel system to be registered as a "
                                                              "fitted channel system, but it was not found")

    # Check if the channel split property is a tuple
    ch_splits = rbp_module.channel_splits[channel_system_name]
    assert isinstance(ch_splits, tuple), (f"Expected channel splits of the channel system to be a tuple, but found "
                                          f"{type(rbp_module.channel_splits[channel_system_name])}")

    # Check if the channel split property has the same length as number of pooling modules
    assert len(ch_splits) == len(pooling_methods), (f"Expected number of channel splits to be the same as the "
                                                    f"number of pooling modules, but found {len(ch_splits)} and "
                                                    f"{len(pooling_methods)}")

    # Check if the values of the channel split property have correct types
    assert all(isinstance(chs_in_regs, ChannelsInRegionSplit) for chs_in_regs in ch_splits), \
        (f"Expected all elements to be of type {ChannelsInRegionSplit.__name__}, but found "
         f"{set(type(chs_in_regs) for chs_in_regs in ch_splits)}")

    # Check if all electrodes are assigned a region, in all channel splits
    channel_names = montage.ch_names
    for ch_split in ch_splits:
        # Loop though all electrodes
        for ch_name in channel_names:
            assert any(ch_name in region.ch_names for region in ch_split.ch_names.values()), \
                f"Expected the channel '{ch_name}' to be contained in a region, but found no match"

    # Check if number of regions in each channel split is as expected
    for expected_regions, ch_split in zip(num_regions, ch_splits):
        assert expected_regions == len(ch_split), (f"Expected number of regions to be {expected_regions}, but found "
                                                   f"{len(ch_split)}")


def test_single_cs_forward():
    """Test the forward method. That it runs, and that output types and shapes are as expected"""
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
    # Make RBP object and fit channel system (tested above)
    # ----------------
    # Hyperparameters
    num_regions = 11, 7, 26, 18

    pooling_methods = ("SingleCSSharedRocket", "SingleCSMean", "SingleCSMean", "SingleCSSharedRocket")
    pooling_methods_kwargs = ({"num_regions": num_regions[0], "num_kernels": 300, "max_receptive_field": 73},
                              {},
                              {},
                              {"num_regions": num_regions[3], "num_kernels": 23, "max_receptive_field": 131})
    split_methods = ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit", "VoronoiSplit")
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    split_methods_kwargs = ({"num_points": num_regions[0], **box_params}, {"num_points": num_regions[1], **box_params},
                            {"num_points": num_regions[2], **box_params}, {"num_points": num_regions[3], **box_params})

    # Make object
    rbp_module = SingleChannelSplitRegionBasedPooling(pooling_methods=pooling_methods,
                                                      pooling_methods_kwargs=pooling_methods_kwargs,
                                                      split_methods=split_methods,
                                                      split_methods_kwargs=split_methods_kwargs)

    # Fit channel system
    rbp_module.fit_channel_system(channel_system=channel_system)

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = rbp_module.pre_compute(data)
    outputs = rbp_module(data, channel_system_name=channel_system_name, channel_name_to_index=channel_name_to_index,
                         pre_computed=pre_computed)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, tuple), f"Expected outputs to be a tuple, but found {type(outputs)}"

    # Check that all elements are torch tensors
    assert all(isinstance(out, torch.Tensor) for out in outputs)

    # Check if the sizes are correct
    assert all(out.size() == torch.Size([batch_size, expected_regions, time_steps])
               for out, expected_regions in zip(outputs, num_regions))

    # todo: check if any tensor is empty or not


# ------------------------
# Testing MultiChannelSplitRegionBasedPooling
# ------------------------
def test_multi_cs_fit_channel_system():
    """Tests if a channel system is properly fit. Testing channel_split property types, lengths, that the electrodes
    are contained in regions when expected, and that the number of regions are correct"""
    # ----------------
    # Generate dummy channel system
    # todo: check mCoding if this can be improved
    # ----------------
    channel_system_name = "TestName"
    x_min, x_max, y_min, y_max = -.17, .17, -.17, .17
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    electrode_positions = Electrodes3D(montage.get_positions()["ch_pos"])
    channel_name_to_index = {name: i for i, name in enumerate(electrode_positions.positions)}
    channel_system = ChannelSystem(name=channel_system_name, channel_name_to_index=channel_name_to_index,
                                   electrode_positions=electrode_positions)

    # ----------------
    # Make RBP object
    # todo: check mCoding if this can be improved
    # ----------------
    num_regions = (11, 7, 26, 4, 7)
    num_channel_splits = len(num_regions)

    pooling_method = "MultiCSSharedRocket"
    pooling_method_kwargs = {"num_regions": num_regions, "num_kernels": 43, "max_receptive_field": 37}
    split_methods = tuple("VoronoiSplit" for _ in range(num_channel_splits))
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    split_methods_kwargs = tuple({"num_points": num_regs, **box_params} for num_regs in num_regions)

    rbp_module = MultiChannelSplitsRegionBasedPooling(pooling_method=pooling_method,
                                                      pooling_method_kwargs=pooling_method_kwargs,
                                                      split_methods=split_methods,
                                                      split_methods_kwargs=split_methods_kwargs)

    # ----------------
    # Fit channel system
    # ----------------
    rbp_module.fit_channel_system(channel_system=channel_system)

    # ----------------
    # Tests  todo: very overlapping with the Single CS version
    # ----------------
    # Check if the channel system is a key in the registered channel splits
    assert channel_system_name in rbp_module.channel_splits, ("Expected the channel system to be registered as a "
                                                              "fitted channel system, but it was not found")

    # Check if the channel split property is a tuple
    ch_splits = rbp_module.channel_splits[channel_system_name]
    assert isinstance(ch_splits, tuple), \
        (f"Expected channel splits of the channel system to be a tuple, but found "
         f"{type(rbp_module.channel_splits[channel_system_name])}")

    # Check if the channel split property (constrained to the specific channel system) has the same length as number of
    # pooling modules
    assert len(ch_splits) == len(split_methods), (f"Expected number of channel splits to be {len(split_methods)}, but "
                                                  f"found {len(ch_splits)}")

    # Check if the values of the channel split property have correct types
    assert all(isinstance(chs_in_regs, ChannelsInRegionSplit) for chs_in_regs in ch_splits), \
        (f"Expected all elements to be of type {ChannelsInRegionSplit.__name__}, but found "
         f"{set(type(chs_in_regs) for chs_in_regs in ch_splits)}")

    # Check if all electrodes are assigned a region, in all channel splits
    channel_names = montage.ch_names
    for ch_split in ch_splits:
        # Loop though all electrodes
        for ch_name in channel_names:
            assert any(ch_name in region.ch_names for region in ch_split.ch_names.values()), \
                f"Expected the channel '{ch_name}' to be contained in a region, but found no match"

    # Check if number of regions in each channel split is as expected
    for expected_regions, ch_split in zip(num_regions, ch_splits):
        assert expected_regions == len(ch_split), (f"Expected number of regions to be {expected_regions}, but found "
                                                   f"{len(ch_split)}")


def test_multi_cs_forward():
    """Test the forward method. That it runs, and that output types and shapes are as expected"""
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
    # Make RBP object and fit channel system (tested above)
    # ----------------
    # Hyperparameters
    num_regions = (11, 7, 26, 4, 7)
    num_channel_splits = len(num_regions)

    pooling_method = "MultiCSSharedRocket"
    pooling_method_kwargs = {"num_regions": num_regions, "num_kernels": 43, "max_receptive_field": 37}
    split_methods = tuple("VoronoiSplit" for _ in range(num_channel_splits))
    box_params = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    split_methods_kwargs = tuple({"num_points": num_regs, **box_params} for num_regs in num_regions)

    # Make object
    rbp_module = MultiChannelSplitsRegionBasedPooling(pooling_method=pooling_method,
                                                      pooling_method_kwargs=pooling_method_kwargs,
                                                      split_methods=split_methods,
                                                      split_methods_kwargs=split_methods_kwargs)

    # Fit channel system
    rbp_module.fit_channel_system(channel_system=channel_system)

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = rbp_module.pre_compute(data)
    outputs = rbp_module(data, channel_system_name=channel_system_name, channel_name_to_index=channel_name_to_index,
                         pre_computed=pre_computed)

    # ----------------
    # Tests
    # ----------------
    # Type check
    assert isinstance(outputs, tuple), f"Expected outputs to be a tuple, but found {type(outputs)}"

    # Check that all elements are torch tensors
    assert all(isinstance(out, torch.Tensor) for out in outputs)

    # Check if the sizes are correct
    assert all(out.size() == torch.Size([batch_size, expected_channel_dim, time_steps])
               for out, expected_channel_dim in zip(outputs, num_regions))


# ------------------------
# Testing the main RBP module
# ------------------------
def test_main_forward():
    """Test forward method. This it runs, has correct output type, and correct output shapes"""
    # ----------------
    # Generate dummy channel system
    # todo: re-using code
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
    num_channel_splits_2 = len(num_regions_2)

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
    # Make RBP module
    # ----------------
    # Create model
    rbp_module = RegionBasedPooling((design_1, design_2, design_3))

    # Fit channel system
    rbp_module.fit_channel_system(channel_system)

    # ----------------
    # Pre-compute and run forward method
    # ----------------
    pre_computed = rbp_module.pre_compute(data)
    outputs = rbp_module(data, channel_system_name=channel_system_name, channel_name_to_index=channel_name_to_index,
                         pre_computed=pre_computed)

    # ----------------
    # Tests  todo: very overlapping with the above tests of forward method
    # ----------------
    # Type check
    assert isinstance(outputs, tuple), f"Expected outputs to be a tuple, but found {type(outputs)}"

    # Check if the number of elements are as expected (=total number of channel/region splits)
    tot_num_channel_splits = num_channel_splits_1 + num_channel_splits_2 + num_channel_splits_3
    assert len(outputs) == tot_num_channel_splits, (f"Expected output tuple length to match with the total number of "
                                                    f"channel/region splits, ({tot_num_channel_splits}), but found "
                                                    f"{len(outputs)}")

    # Check that all elements are torch tensors
    assert all(isinstance(out, torch.Tensor) for out in outputs)

    # Check if the sizes of tensors are correct
    expected_channel_dims = num_regions_1 + num_regions_2 + num_regions_3
    assert all(out.size() == torch.Size([batch_size, expected_channel_dim, time_steps])
               for out, expected_channel_dim in zip(outputs, expected_channel_dims))
