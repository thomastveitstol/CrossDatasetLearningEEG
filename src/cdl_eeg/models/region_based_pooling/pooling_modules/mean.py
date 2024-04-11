from typing import List

import torch

from cdl_eeg.data.datasets.dataset_base import channel_names_to_indices
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import MultiMontageSplitsPoolingBase


class MultiMSMean(MultiMontageSplitsPoolingBase):
    """
    Pooling by computing average in channel dimension

    While this one does not actually require pre-computing (it can be done, but my experience is that the increase in
    memory consumption is not worth the decrease in run time)

    Examples
    --------
    >>> _ = MultiMSMean()
    >>> MultiMSMean.supports_precomputing()
    False
    """

    @staticmethod
    def forward(input_tensors, *, channel_splits, channel_name_to_index):
        """
        Forward method

        (unittest in test folder)

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
            A dict with keys being dataset names and values are tensors containing EEG data with
            shape=(batch, channels, time_steps). Note that the channels are correctly selected within this method, and
            the EEG data should be the full data matrix (such that channel_name_to_index maps correctly)
        channel_splits : dict[str, cdl_eeg.models.region_based_pooling.utils.CHANNELS_IN_MONTAGE_SPLIT]
        channel_name_to_index : dict[str, dict[str, int]]

        Returns
        -------
        tuple[torch.Tensor, ...]
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
                _forward_single_dataset(x, channel_splits=ch_splits, channel_name_to_index=dataset_ch_name_to_idx))

        # Concatenate the data together
        return tuple(torch.cat(tensor, dim=0) for tensor in list(zip(*dataset_region_representations)))


# -----------------
# Functions
# -----------------
def _forward_single_dataset(x, *, channel_splits, channel_name_to_index):
    # --------------
    # Loop through all channel/region splits
    # --------------
    output_channel_splits: List[torch.Tensor] = []
    for channel_split in channel_splits:
        # Initialise tensor which will contain all region representations
        batch, _, time_steps = x.size()
        num_regions = len(channel_split)
        region_representations = torch.empty(size=(batch, num_regions, time_steps)).to(x.device)

        # Loop through all regions (dicts are ordered by default, meaning the ordering of the tensor will follow the
        # ordering of the dict keys)
        for i, channels in enumerate(channel_split.values()):
            # Extract the indices of the legal channels for this region
            allowed_node_indices = channel_names_to_indices(ch_names=channels,
                                                            channel_name_to_index=channel_name_to_index)

            # Compute region representation by averaging and insert it
            region_representations[:, i] = torch.mean(x[:, allowed_node_indices])  # Consider to keep dim in the future

        # Append as montage split output
        output_channel_splits.append(region_representations)

    return output_channel_splits
