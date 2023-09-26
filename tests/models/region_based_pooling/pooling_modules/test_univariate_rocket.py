from typing import Dict, Tuple

import torch

from cdl_eeg.models.region_based_pooling.pooling_modules.univariate_rocket import MultiCSSharedRocket, \
    SingleCSSharedRocket
from cdl_eeg.models.region_based_pooling.utils import ChannelsInRegionSplit, RegionID, ChannelsInRegion


def test_single_cs_shared_rocket_forward():
    """Test forward method of SingleCSSharedRocket. Tests output type and shape"""
    # ---------------
    # Define model
    # ---------------
    # Hyperparameters
    num_regions = 7
    num_kernels = 100
    max_receptive_field = 48

    # Define model
    model = SingleCSSharedRocket(num_regions, num_kernels=num_kernels, max_receptive_field=max_receptive_field)

    # ---------------
    # Prepare inputs to forward method
    # ---------------
    # Create dummy data  todo: improve dummy data usage
    channel_split = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("A", "C", "E")),
                                           RegionID(1): ChannelsInRegion(("C", "E", "B", "A")),
                                           RegionID(2): ChannelsInRegion(("F",)),
                                           RegionID(3): ChannelsInRegion(("C", "B")),
                                           RegionID(4): ChannelsInRegion(("F", "E", "B", "A", "C")),
                                           RegionID(5): ChannelsInRegion(("D", "C", "B", "A")),
                                           RegionID(6): ChannelsInRegion(("C", "E", "B", "A", "F", "D"))})
    channel_name_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    batch_size, time_steps = 10, 2000
    x = torch.rand(batch_size, len(channel_name_to_index), time_steps)

    # ---------------
    # Pre-compute and run forward method
    # ---------------
    pre_computed = model.pre_compute(x)
    outputs = model(x, pre_computed=pre_computed, channel_split=channel_split,
                    channel_name_to_index=channel_name_to_index)

    # ---------------
    # Tests
    # todo: test that the matrix multiplication is safe and as expected
    # ---------------
    # Type check
    assert isinstance(outputs, torch.Tensor), f"Expected output to be a torch.Tensor, but found {type(outputs)}"

    # Shape check
    assert outputs.size() == torch.Size([batch_size, num_regions, time_steps]), \
        f"Expected size={torch.Size([batch_size, num_regions, time_steps])}, but found size={outputs.size()}"


def test_multi_cs_shared_rocket_forward():
    """Test forward method of MultiCSSharedRocket. Tests output type and shape"""
    # ---------------
    # Define model
    # ---------------
    # Hyperparameters
    num_regions = (4, 7, 3)
    num_kernels = 100
    max_receptive_field = 48

    # Define model
    model = MultiCSSharedRocket(num_regions, num_kernels=num_kernels, max_receptive_field=max_receptive_field)

    # ---------------
    # Prepare inputs to forward method
    # ---------------
    channel_splits: Dict[str, Tuple[ChannelsInRegionSplit, ...]] = dict()
    channel_name_to_index: Dict[str, Dict[str, int]] = dict()

    # First dataset  todo: improve dummy data usage
    channel_split_0 = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("E", "C", "B")),
                                             RegionID(1): ChannelsInRegion(("E", "F", "A")),
                                             RegionID(2): ChannelsInRegion(("F", "A", "B", "C")),
                                             RegionID(3): ChannelsInRegion(("C", "B"))})
    channel_split_1 = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("A", "C", "E")),
                                             RegionID(1): ChannelsInRegion(("C", "E", "B", "A")),
                                             RegionID(2): ChannelsInRegion(("F",)),
                                             RegionID(3): ChannelsInRegion(("C", "B")),
                                             RegionID(4): ChannelsInRegion(("F", "E", "B", "A", "C")),
                                             RegionID(5): ChannelsInRegion(("D", "C", "B", "A")),
                                             RegionID(6): ChannelsInRegion(("C", "E", "B", "A", "F", "D"))})
    channel_split_2 = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("B", "F")),
                                             RegionID(1): ChannelsInRegion(("B", "A", "C")),
                                             RegionID(2): ChannelsInRegion(("A",))})

    channel_splits["d1"] = (channel_split_0, channel_split_1, channel_split_2)
    channel_name_to_index["d1"] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    # Second dataset
    channel_split_0 = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("q", "w", "e")),
                                             RegionID(1): ChannelsInRegion(("r",)),
                                             RegionID(2): ChannelsInRegion(("t", "r", "e", "b")),
                                             RegionID(3): ChannelsInRegion(("h", "j"))})
    channel_split_1 = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("c", "t", "q", "b")),
                                             RegionID(1): ChannelsInRegion(("q", "c")),
                                             RegionID(2): ChannelsInRegion(("s",)),
                                             RegionID(3): ChannelsInRegion(("h",)),
                                             RegionID(4): ChannelsInRegion(("j", "e", "r", "k", "j", "w", "q")),
                                             RegionID(5): ChannelsInRegion(("e", "r", "b", "a")),
                                             RegionID(6): ChannelsInRegion(("s", "q", "w"))})
    channel_split_2 = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("e",)),
                                             RegionID(1): ChannelsInRegion(("r",)),
                                             RegionID(2): ChannelsInRegion(("l",))})

    channel_splits["d2"] = (channel_split_0, channel_split_1, channel_split_2)
    channel_name_to_index["d2"] = {"q": 0, "w": 1, "e": 2, "r": 3, "t": 4, "b": 5, "h": 6, "j": 7, "c": 8, "s": 9,
                                   "k": 10, "l": 11, "a": 12}

    # ---------------
    # Pre-compute and run forward method
    # ---------------
    time_steps = 2000
    input_tensors = {"d1": torch.rand(size=(13, len(channel_name_to_index["d1"]), time_steps)),
                     "d2": torch.rand(size=(31, len(channel_name_to_index["d2"]), time_steps))}

    pre_computed = model.pre_compute(input_tensors)
    outputs = model(input_tensors, pre_computed=pre_computed, channel_splits=channel_splits,
                    channel_name_to_index=channel_name_to_index)

    # ---------------
    # Tests
    # todo: test if the batches will be stacked as expected
    # todo: test that the matrix multiplication is safe and as expected
    # ---------------
    # Type check
    assert isinstance(outputs, tuple), f"Expected output to be a tuple, but found {type(outputs)}"

    # Check that all elements are torch tensors
    assert all(isinstance(out, torch.Tensor) for out in outputs)

    # Check if the sizes are correct
    expected_batch_size = 31 + 13
    assert all(out.size() == torch.Size([expected_batch_size, expected_regions, time_steps])
               for out, expected_regions in zip(outputs, num_regions))
