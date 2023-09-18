"""
Classes for Region Based Pooling

There will likely be some overlap with a former project (where Region Based Pooling was first introduced) at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/models/modules/bins/regions_to_bins.py
"""
import abc
from typing import Dict, Tuple

import torch.nn as nn

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.region_based_pooling.pooling_modules.getter import get_pooling_module
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import SingleChannelSplitPoolingBase
from cdl_eeg.models.region_based_pooling.region_splits.getter import get_region_split
from cdl_eeg.models.region_based_pooling.utils import ChannelsInRegionSplit


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
    >>> my_split_kwargs = ({"num_points": 7, "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
    ...                    {"num_points": 11, "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
    >>> _ = SingleChannelRegionBasedPooling(pooling_methods=("SingleCSMean", "SingleCSMean"),
    ...                                     pooling_methods_kwargs=({}, {}),
    ...                                     split_methods=("VoronoiSplit", "VoronoiSplit"),
    ...                                     split_methods_kwargs=my_split_kwargs)
    """

    def __init__(self, pooling_methods, pooling_methods_kwargs, split_methods, split_methods_kwargs):
        """
        Initialise

        Parameters
        ----------
        pooling_methods : tuple[str, ...]
            Pooling methods
        pooling_methods_kwargs : tuple[dict[str, typing.Any], ...]
            Keyword arguments of the pooling modules. Must have the same length as pooling_methods.
        split_methods : tuple[str, ...]
            Region splits method. Must have the same length as pooling_methods.
        split_methods_kwargs : tuple[dict[str, typing.Any], ...]
            Keyword arguments of the pooling modules. Must have the same length as split_methods.
        """
        super().__init__()

        # -------------------
        # Input checks
        # -------------------
        self._input_checks(pooling_methods, pooling_methods_kwargs, split_methods, split_methods_kwargs)

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
            raise TypeError(f"Expected all pooling methods to inherit from {SingleChannelSplitPoolingBase.__name__}, "
                            f"but found {wrong_methods}")

        # Pass them to nn.ModuleList (otherwise they are not registered as modules with parameters py pytorch)
        self._pooling_modules = nn.ModuleList(pooling_modules)

        # -------------------
        # Region splits
        # -------------------
        # tuple[cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit, ...]
        # Generate and store region splits
        self._region_splits = tuple(get_region_split(split_method, **kwargs)
                                    for split_method, kwargs in zip(split_methods, split_methods_kwargs))

        # Initialise the mapping from regions to channel names, for all datasets (must be fit later)
        # Should be {dataset_name: tuple[ChannelsInRegionSplit, ...]}
        self._channel_splits: Dict[str, Tuple[ChannelsInRegionSplit, ...]] = dict()

    @staticmethod
    def _input_checks(pooling_methods, pooling_methods_kwargs, split_methods, split_methods_kwargs):
        # Check if the pooling methods, pooling method kwargs, split_methods, and split_methods_kwargs have the same
        # lengths
        expected_same_lengths = (pooling_methods, pooling_methods_kwargs, split_methods, split_methods_kwargs)
        if not all(len(arg) == len(pooling_methods) for arg in expected_same_lengths):
            raise ValueError(f"Expected pooling methods, their corresponding kwargs, and the region splits methods and "
                             f"their corresponding kwargs all to have the same lengths, but found lengths "
                             f"{(tuple(len(arg) for arg in expected_same_lengths))}")

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

    # ----------------
    # Methods for fitting channel systems
    # todo: I think these might be moved to the base class
    # ----------------
    def fit_channel_system(self, channel_system):
        """
        Fit a single channel system on the regions splits

        (unit test in test folder)
        Parameters
        ----------
        channel_system : cdl_eeg.data.datasets.dataset_base.ChannelSystem
            The channel system to fit
        Returns
        -------
        None
        """
        self._channel_splits[channel_system.name] = tuple(
            region_split.place_in_regions(channel_system.electrode_positions) for region_split in self._region_splits)

    def fit_channel_systems(self, channel_systems):
        """
        Fit multiple channel systems on the regions splits

        Parameters
        ----------
        channel_systems : tuple[ChannelSystem, ...] | ChannelSystem
            Channel systems to fit

        Returns
        -------
        None
        """
        # Run the .fit_channel_system on all channel systems passed
        if isinstance(channel_systems, ChannelSystem):
            self.fit_channel_system(channel_system=channel_systems)
        elif isinstance(channel_systems, tuple) and all(isinstance(ch_system, ChannelSystem)
                                                        for ch_system in channel_systems):
            for channel_system in channel_systems:
                self.fit_channel_system(channel_system=channel_system)
        else:
            raise TypeError(f"Expected channel systems to be either a channel system (type={ChannelSystem.__name__}) "
                            f"or a tuple of channel systems, but this was not the case")

    # ----------------
    # Properties
    # ----------------
    @property
    def channel_splits(self):
        return self._channel_splits
