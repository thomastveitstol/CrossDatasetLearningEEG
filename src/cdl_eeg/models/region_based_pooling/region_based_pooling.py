"""
Classes for Region Based Pooling

There will likely be some overlap with a former project (where Region Based Pooling was first introduced) at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/models/modules/bins/regions_to_bins.py
"""
import abc

import torch.nn as nn

from cdl_eeg.models.region_based_pooling.pooling_modules.getter import get_pooling_module
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import SingleChannelSplitPoolingBase


# ------------------
# Base class
# ------------------
class RegionBasedPoolingBase(nn.Module, abc.ABC):
    """
    Base class for all Region Based Pooling classes
    """


# ------------------
# Implementations of RBP
# ------------------
class SingleChannelRegionBasedPooling(RegionBasedPoolingBase):
    """
    Region Based Pooling when pooling module operates on a single channel split at once (when the pooling module used
    inherits from SingleChannelSplitPoolingBase)

    Examples
    --------
    >>> _ = SingleChannelRegionBasedPooling(pooling_methods=("SingleCSMean", "SingleCSMean"),
    ...                                     pooling_methods_kwargs=({}, {}))
    """

    def __init__(self, pooling_methods, pooling_methods_kwargs):
        """
        Initialise

        Parameters
        ----------
        pooling_methods : tuple[str, ...]
            Pooling methods
        pooling_methods_kwargs : tuple[dict[str, typing.Any], ...]
            Keyword arguments of the pooling modules. Must have the same length as pooling_methods.
        """
        super().__init__()

        # -------------------
        # Input checks
        # -------------------
        self._input_checks(pooling_methods, pooling_methods_kwargs)

        # -------------------
        # Generate pooling modules
        # -------------------
        # Get them in a list
        pooling_modules = [get_pooling_module(pooling_method, **kwargs) for pooling_method, kwargs
                           in zip(pooling_methods, pooling_methods_kwargs)]

        # Verify that they have correct type
        if not all(isinstance(pooling_module, SingleChannelSplitPoolingBase) for pooling_module in pooling_modules):
            wrong_methods = tuple(pooling_method for pooling_method in pooling_methods
                                  if not isinstance(pooling_method, SingleChannelSplitPoolingBase))
            raise TypeError(f"Expected all pooling methods to inherit from {SingleChannelSplitPoolingBase}, but found "
                            f"{wrong_methods}")

        # Pass them to nn.ModuleList (otherwise they are not registered as modules with parameters py pytorch)
        self._pooling_modules = nn.ModuleList(pooling_modules)

    @staticmethod
    def _input_checks(pooling_methods, pooling_methods_kwargs):
        # Check if the pooling methods and pooling method kwargs have same length
        if len(pooling_methods) != len(pooling_methods_kwargs):
            raise ValueError(f"Expected pooling methods to have the same length as their corresponding kwargs, but "
                             f"found lengths {len(pooling_methods)} and {len(pooling_methods_kwargs)}")
        # TODO: Must ensure that all pooling methods are modules/objects from SingleChannelSplitPoolingBase

    def forward(self, x, *, channel_splits, channel_name_to_index):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        channel_splits : tuple[cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit, ...]
        channel_name_to_index : dict[str, int]

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        # Pass through all channel splits and return as tuple
        return tuple(pooling_module(x, channel_splits=channel_splits, channel_name_to_index=channel_name_to_index)
                     for pooling_module in self._pooling_modules)
