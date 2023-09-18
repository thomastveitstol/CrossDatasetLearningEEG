import torch

from cdl_eeg.data.datasets.dataset_base import channel_names_to_indices
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import SingleChannelSplitPoolingBase


class SingleCSMean(SingleChannelSplitPoolingBase):
    """
    Pooling by computing average in channel dimension. It operates on an entire channel/region split

    Examples
    --------
    >>> _ = SingleCSMean()
    """

    @staticmethod
    def forward(x, *, channel_split, channel_name_to_index):
        """
        Forward method

        TODO: I must implement padding/masking or something like that to enable varied number of channels in the same
         batch

        Parameters
        ----------
        x : torch.Tensor
            A tensor containing EEG data with shape=(batch, channels, time_steps). Note that the channels are correctly
            selected within this method, and the EEG data should be the full data matrix (such that
            channel_name_to_index maps correctly)
        channel_split : cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit
        channel_name_to_index : dict[str, int]

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> from cdl_eeg.models.region_based_pooling.utils import ChannelsInRegionSplit, RegionID, ChannelsInRegion
        >>> my_channels = ChannelsInRegionSplit({RegionID(0): ChannelsInRegion(("A", "C", "E")),
        ...                                      RegionID(1): ChannelsInRegion(("C", "E", "B"))})
        >>> name_2_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
        >>> my_model = SingleCSMean()
        >>> my_model(torch.rand(size=(10, 6, 1000)), channel_split=my_channels, channel_name_to_index=name_2_idx).size()
        torch.Size([10, 2, 1000])
        """
        # Initialise tensor which will contain all region representations
        batch, _, time_steps = x.size()
        num_regions = len(channel_split)
        region_representations = torch.empty(size=(batch, num_regions, time_steps)).to(x.device)

        # Loop through all regions (dicts are ordered by default, meaning the ordering of the tensor will follow the
        # ordering of the dict keys)
        for i, channels in enumerate(channel_split.ch_names.values()):
            # Extract the indices of the legal channels for this region
            allowed_node_indices = channel_names_to_indices(ch_names=channels.ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # Compute region representation by averaging and insert it
            region_representations[:, i] = torch.mean(x[:, allowed_node_indices])  # Consider to keep dim in the future

        return region_representations
