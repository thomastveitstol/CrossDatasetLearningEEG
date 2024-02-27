import torch

from cdl_eeg.data.datasets.dataset_base import channel_names_to_indices
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import SingleChannelSplitPoolingBase


class SingleCSMean(SingleChannelSplitPoolingBase):
    """
    Pooling by computing average in channel dimension. It operates on an entire channel/region split

    Examples
    --------
    >>> _ = SingleCSMean()
    >>> SingleCSMean.supports_precomputing()
    False
    """

    @staticmethod
    def forward(input_tensors, *, channel_splits, channel_name_to_index):
        """
        Forward method

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
            A dict with keys being dataset names and values are tensors containing EEG data with
            shape=(batch, channels, time_steps). Note that the channels are correctly selected within this method, and
            the EEG data should be the full data matrix (such that channel_name_to_index maps correctly)
        channel_splits : dict[str, cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit]
        channel_name_to_index : dict[str, dict[str, int]]

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> from cdl_eeg.models.region_based_pooling.utils import ChannelsInRegionSplit, RegionID
        >>> my_channels_a = ChannelsInRegionSplit({RegionID(0): ("A", "C", "E"),
        ...                                        RegionID(1): ("C", "E", "B", "A")})
        >>> my_channels_b = ChannelsInRegionSplit({RegionID(0): ("E1", "E3", "E2"),
        ...                                        RegionID(1): ("E2",)})
        >>> name_2_idx = {"a": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}, "b": {"E1": 0, "E2": 1, "E3": 2}}
        >>> my_model = SingleCSMean()
        >>> my_model({"a": torch.rand(size=(10, 6, 1000)), "b": torch.rand(size=(7, 3, 1000))},
        ...          channel_splits={"a": my_channels_a, "b": my_channels_b}, channel_name_to_index=name_2_idx).size()
        torch.Size([17, 2, 1000])
        """
        # TODO: I think this may be move to base class?
        # Loop through all datasets
        dataset_region_representations = []
        for dataset_name, x in input_tensors.items():
            # (I take no chances on all input being ordered similarly)
            dataset_ch_name_to_idx = channel_name_to_index[dataset_name]
            ch_splits = channel_splits[dataset_name]

            # Perform forward pass
            dataset_region_representations.append(
                _forward_single_dataset(x, channel_split=ch_splits, channel_name_to_index=dataset_ch_name_to_idx))

        # Concatenate the data together
        return torch.cat(dataset_region_representations, dim=0)


# -----------------
# Functions
# -----------------
def _forward_single_dataset(x, *, channel_split, channel_name_to_index):
    # Initialise tensor which will contain all region representations
    batch, _, time_steps = x.size()
    num_regions = len(channel_split)
    region_representations = torch.empty(size=(batch, num_regions, time_steps)).to(x.device)

    # Loop through all regions (dicts are ordered by default, meaning the ordering of the tensor will follow the
    # ordering of the dict keys)
    for i, channels in enumerate(channel_split.ch_names.values()):
        # Extract the indices of the legal channels for this region
        allowed_node_indices = channel_names_to_indices(ch_names=channels, channel_name_to_index=channel_name_to_index)

        # Compute region representation by averaging and insert it
        region_representations[:, i] = torch.mean(x[:, allowed_node_indices])  # Consider to keep dim in the future

    return region_representations
