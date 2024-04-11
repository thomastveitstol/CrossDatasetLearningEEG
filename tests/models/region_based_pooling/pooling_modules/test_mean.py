from typing import Dict, Tuple

import torch

from cdl_eeg.models.region_based_pooling.pooling_modules.mean import MultiMSMean
from cdl_eeg.models.region_based_pooling.utils import RegionID, CHANNELS_IN_MONTAGE_SPLIT


def test_multi_ms_mean_forward():
    """Test forward method of MultiMSMean. Tests output type and shape"""
    # ---------------
    # Define model
    # ---------------
    # Hyperparameters
    num_regions = (4, 7, 3)

    # Define model
    model = MultiMSMean(num_regions=num_regions)

    # ---------------
    # Prepare inputs to forward method
    # ---------------
    channel_splits: Dict[str, Tuple[CHANNELS_IN_MONTAGE_SPLIT, ...]] = dict()
    channel_name_to_index: Dict[str, Dict[str, int]] = dict()

    # First dataset  todo: improve dummy data usage
    channel_split_0 = {RegionID(0): ("E", "C", "B"),
                       RegionID(1): ("E", "F", "A"),
                       RegionID(2): ("F", "A", "B", "C"),
                       RegionID(3): ("C", "B")}
    channel_split_1 = {RegionID(0): ("A", "C", "E"),
                       RegionID(1): ("C", "E", "B", "A"),
                       RegionID(2): ("F",),
                       RegionID(3): ("C", "B"),
                       RegionID(4): ("F", "E", "B", "A", "C"),
                       RegionID(5): ("D", "C", "B", "A"),
                       RegionID(6): ("C", "E", "B", "A", "F", "D")}
    channel_split_2 = {RegionID(0): ("B", "F"),
                       RegionID(1): ("B", "A", "C"),
                       RegionID(2): ("A",)}

    channel_splits["d1"] = (channel_split_0, channel_split_1, channel_split_2)
    channel_name_to_index["d1"] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    # Second dataset
    channel_split_0 = {RegionID(0): ("q", "w", "e"),
                       RegionID(1): ("r",),
                       RegionID(2): ("t", "r", "e", "b"),
                       RegionID(3): ("h", "j")}
    channel_split_1 = {RegionID(0): ("c", "t", "q", "b"),
                       RegionID(1): ("q", "c"),
                       RegionID(2): ("s",),
                       RegionID(3): ("h",),
                       RegionID(4): ("j", "e", "r", "k", "j", "w", "q"),
                       RegionID(5): ("e", "r", "b", "a"),
                       RegionID(6): ("s", "q", "w")}
    channel_split_2 = {RegionID(0): ("e",),
                       RegionID(1): ("r",),
                       RegionID(2): ("l",)}

    channel_splits["d2"] = (channel_split_0, channel_split_1, channel_split_2)
    channel_name_to_index["d2"] = {"q": 0, "w": 1, "e": 2, "r": 3, "t": 4, "b": 5, "h": 6, "j": 7, "c": 8, "s": 9,
                                   "k": 10, "l": 11, "a": 12}

    # ---------------
    # Run forward method
    # ---------------
    time_steps = 2000
    input_tensors = {"d1": torch.rand(size=(13, len(channel_name_to_index["d1"]), time_steps)),
                     "d2": torch.rand(size=(31, len(channel_name_to_index["d2"]), time_steps))}

    outputs = model(input_tensors, channel_splits=channel_splits, channel_name_to_index=channel_name_to_index)

    # ---------------
    # Tests
    # todo: test if the batches will be stacked as expected
    # ---------------
    # Type check
    assert isinstance(outputs, tuple), f"Expected output to be a tuple, but found {type(outputs)}"

    # Check that all elements are torch tensors
    assert all(isinstance(out, torch.Tensor) for out in outputs)

    # Check if the sizes are correct
    expected_batch_size = 31 + 13
    assert all(out.size() == torch.Size([expected_batch_size, expected_regions, time_steps])
               for out, expected_regions in zip(outputs, num_regions))
