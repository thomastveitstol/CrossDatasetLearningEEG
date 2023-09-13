from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import SingleChannelSplitPoolingBase, \
    MultiChannelSplitsPoolingBase


class SingleCSMean(SingleChannelSplitPoolingBase):
    ...


class MultiCSMean(MultiChannelSplitsPoolingBase):

    def forward(self, x, channel_split, channel_name_to_index):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        channel_split : cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit
        channel_name_to_index : dict[str, int]

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        raise NotImplementedError
