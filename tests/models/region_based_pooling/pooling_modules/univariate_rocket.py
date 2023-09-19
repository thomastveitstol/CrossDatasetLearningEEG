import torch

from cdl_eeg.models.region_based_pooling.pooling_modules.univariate_rocket import SingleCSSharedRocket
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
