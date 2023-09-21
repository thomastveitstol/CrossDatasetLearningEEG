"""
Classes for Region Based Pooling

There will likely be some overlap with a former project (where Region Based Pooling was first introduced) at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/models/modules/bins/regions_to_bins.py

Authored by:
    Thomas Tveitstøl (Oslo University Hospital)
"""
import abc
import dataclasses
import itertools
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.region_based_pooling.pooling_modules.getter import get_pooling_module
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import SingleChannelSplitPoolingBase, \
    MultiChannelSplitsPoolingBase
from cdl_eeg.models.region_based_pooling.region_splits.getter import get_region_split
from cdl_eeg.models.region_based_pooling.region_splits.region_split_base import RegionSplitBase
from cdl_eeg.models.region_based_pooling.utils import ChannelsInRegionSplit


# ------------------
# Convenient dataclass
# ------------------
class RBPPoolType(Enum):
    """
    Examples
    --------
    >>> RBPPoolType("single_cs") == RBPPoolType.SINGLE_CS
    True
    """
    SINGLE_CS = "single_cs"
    MULTI_CS = "multi_cs"


@dataclasses.dataclass(frozen=True)
class RBPDesign:
    """Dataclass for creating input to RBP"""
    pooling_methods: Union[str, Tuple[str, ...]]
    pooling_methods_kwargs: Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]
    pooling_type: RBPPoolType
    split_methods: Union[str, Tuple[str, ...]]
    split_methods_kwargs: Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]


# ------------------
# Base class
# ------------------
class RegionBasedPoolingBase(nn.Module, abc.ABC):
    """
    Base class for all Region Based Pooling classes
    """


# ------------------
# Implementations of RBP  todo: should be possible to combine the different classes
# ------------------
class SingleChannelSplitRegionBasedPooling(RegionBasedPoolingBase):
    """
    Region Based Pooling when pooling module operates on a single channel split at once (when the pooling module used
    inherits from SingleChannelSplitPoolingBase)

    Examples
    --------
    >>> my_split_kwargs = ({"num_points": 7, "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
    ...                    {"num_points": 11, "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
    >>> my_model = SingleChannelSplitRegionBasedPooling(pooling_methods=("SingleCSMean", "SingleCSMean"),
    ...                                                 pooling_methods_kwargs=({}, {}),
    ...                                                 split_methods=("VoronoiSplit", "VoronoiSplit"),
    ...                                                 split_methods_kwargs=my_split_kwargs)
    >>> my_model.supports_precomputing
    False

    Check with a pooling module which supports pre-computing

    >>> SingleChannelSplitRegionBasedPooling(pooling_methods=("SingleCSSharedRocket", "SingleCSMean"),
    ...                                      pooling_methods_kwargs=({"num_regions": 7, "num_kernels": 100,
    ...                                                               "max_receptive_field": 200}, {}),
    ...                                      split_methods=("VoronoiSplit", "VoronoiSplit"),
    ...                                      split_methods_kwargs=my_split_kwargs).supports_precomputing
    True
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

    def forward(self, x, *, channel_system_name, channel_name_to_index, pre_computed=None):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        channel_system_name : str
        channel_name_to_index : dict[str, int]
        pre_computed : tuple, optional

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        # ------------------
        # Pass through all channel splits and return as tuple
        # ------------------
        # Simple case when no pre-computing is made
        if not self.supports_precomputing or pre_computed is None:
            return tuple(pooling_module(x, channel_split=channel_split, channel_name_to_index=channel_name_to_index)
                         for pooling_module, channel_split in zip(self._pooling_modules,
                                                                  self._channel_splits[channel_system_name]))

        # Otherwise, append to a list
        output_channel_splits: List[torch.Tensor] = []
        for pre_comp_features, pooling_module, channel_split \
                in zip(pre_computed, self._pooling_modules, self._channel_splits[channel_system_name]):
            # Handle the unsupported case, or when pre-computing is not desired
            if not pooling_module.supports_precomputing() or pre_comp_features is None:
                output_channel_splits.append(pooling_module(x, channel_split=channel_split,
                                                            channel_name_to_index=channel_name_to_index))
            else:
                output_channel_splits.append(pooling_module(x, channel_split=channel_split,
                                                            channel_name_to_index=channel_name_to_index,
                                                            pre_computed=pre_comp_features))
        return tuple(output_channel_splits)

    def pre_compute(self, x):
        """
        Method for pre-computing

        Possible future updates:
            (1) Some pooling modules may have multiple pre-computing methods in the future
            (2) Different pooling modules may require different input arguments
            (3) Improved memory usage?
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of pre-computed tensor (one pr. pooling module). The element will be None if the corresponding
            pooling module does not support pre-computing
        """
        # Quick check to see if this method should be run
        if not self.supports_precomputing:
            raise RuntimeError("Tried to pre-compute when no pooling method supports pre-computing")

        # Loop through all pooling modules
        pre_computed: List[Optional[torch.Tensor]] = []
        for pooling_module in self._pooling_modules:
            if pooling_module.supports_precomputing():
                # Assuming that the method is called 'pre_compute', and that it only takes in 'x' as argument
                pre_computed.append(pooling_module.pre_compute(x))
            else:
                pre_computed.append(None)

        # Convert to tuple and return
        return tuple(pre_computed)

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

    @property
    def supports_precomputing(self):
        return any(pooling_module.supports_precomputing() for pooling_module in self._pooling_modules)


class MultiChannelSplitsRegionBasedPooling(RegionBasedPoolingBase):
    """
    Region Based Pooling when pooling module operates on multiple channel/region split at once (when the pooling
    module used inherits from MultiChannelSplitPoolingBase)

    Examples
    --------
    >>> my_pooling_method = "MultiCSSharedRocket"
    >>> my_pooling_kwargs = {"num_regions": (3, 7, 4), "num_kernels": 43, "max_receptive_field": 37}
    >>> my_split_methods = ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit")
    >>> my_box = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
    >>> my_split_kwargs = ({"num_points": 3, **my_box}, {"num_points": 7, **my_box}, {"num_points": 4, **my_box})
    >>> my_model = MultiChannelSplitsRegionBasedPooling(pooling_method=my_pooling_method,
    ...                                                 pooling_method_kwargs=my_pooling_kwargs,
    ...                                                 split_methods=my_split_methods,
    ...                                                 split_methods_kwargs=my_split_kwargs)
    >>> my_model.supports_precomputing
    True
    """

    def __init__(self, pooling_method, pooling_method_kwargs, split_methods, split_methods_kwargs):
        """
        Initialise

        Parameters
        ----------
        pooling_method : str
            Pooling method
        pooling_method_kwargs : dict[str, typing.Any]
            Keyword arguments of the pooling modules. Must have the same length as pooling_methods.
        split_methods : tuple[str, ...]
            Region split methods
        split_methods_kwargs : tuple[dict[str, typing.Any], ...]
            Keyword arguments of the pooling modules. Must have the same length as split_methods.
        """
        super().__init__()

        # -------------------
        # Input checks
        # -------------------
        self._input_checks(pooling_method_kwargs, split_methods, split_methods_kwargs)

        # -------------------
        # Generate pooling modules
        # -------------------
        # Get correct pooling module in a list
        pooling_module = get_pooling_module(pooling_method, **pooling_method_kwargs)

        # Verify that it has have correct type
        if not isinstance(pooling_module, MultiChannelSplitsPoolingBase):
            raise TypeError(f"Expected all pooling module to inherit from {MultiChannelSplitsPoolingBase.__name__}, "
                            f"but found {type(pooling_module)}")

        # Store pooling module
        self._pooling_module = pooling_module

        # -------------------
        # Channel/Region splits
        # -------------------
        # Generate and store region splits
        self._region_splits = tuple(get_region_split(split_method, **kwargs)
                                    for split_method, kwargs in zip(split_methods, split_methods_kwargs))

        # Initialise the mapping from regions to channel names, for all datasets (must be fit later)
        # Should be {dataset_name: tuple[ChannelsInRegionSplit, ...]}
        self._channel_splits: Dict[str, Tuple[ChannelsInRegionSplit, ...]] = dict()

    # ----------------
    # Checks
    # ----------------
    @staticmethod
    def _input_checks(pooling_method_kwargs, split_methods, split_methods_kwargs):
        # Check if all split methods have corresponding kwargs
        if len(split_methods) != len(split_methods_kwargs):
            raise ValueError(f"Expected number of split methods to be equal to the number of split kwargs, but found "
                             f"{len(split_methods)} and {len(split_methods_kwargs)}")

        # Weak compatibility check with split methods and pooling kwargs
        if "num_regions" in pooling_method_kwargs and len(pooling_method_kwargs["num_regions"]) != len(split_methods):
            raise ValueError(f"Expected number of split methods to be equal to the number of channel/region splits "
                             f"passed to the pooling module, but found {len(split_methods)} and "
                             f"{pooling_method_kwargs['num_regions']}")

    # ----------------
    # Forward and pre-computing
    # ----------------
    def forward(self, x, *, channel_system_name, channel_name_to_index, pre_computed=None):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        channel_system_name : str
        channel_name_to_index : dict[str, int]
        pre_computed : tuple, optional

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        # ------------------
        # Pass through all channel splits and return
        # ------------------
        # Pass pre_computed or not, depending on compatibility and if when pre-computing is not desired (value is None)
        if not self.supports_precomputing or pre_computed is None:
            return self._pooling_module(x, channel_splits=self._channel_splits[channel_system_name],
                                        channel_name_to_index=channel_name_to_index)
        else:
            return self._pooling_module(x, channel_splits=self._channel_splits[channel_system_name],
                                        channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

    def pre_compute(self, x):
        """
        Method for pre-computing

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of pre-computed tensor (one pr. pooling module). The element will be None if the corresponding
            pooling module does not support pre-computing
        """
        # Quick check to see if this method should be run
        if not self.supports_precomputing:
            raise RuntimeError("Tried to pre-compute when no pooling method supports pre-computing")

        # todo: Assuming that the method is called 'pre_compute', and that it only takes in 'x' as argument
        return self._pooling_module.pre_compute(x)

    # ----------------
    # Methods for fitting channel systems
    # todo: I think these might be moved to the base class
    # ----------------
    def fit_channel_system(self, channel_system):
        """
        Fit a single channel system on the regions splits

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
            raise TypeError(
                f"Expected channel systems to be either a channel system (type={ChannelSystem.__name__}) "
                f"or a tuple of channel systems, but this was not the case")

    # ----------------
    # Properties
    # ----------------
    @property
    def channel_splits(self):
        return self._channel_splits

    @property
    def supports_precomputing(self):
        return self._pooling_module.supports_precomputing()


class __DeprecatedMultiChannelSplitsRegionBasedPooling(RegionBasedPoolingBase):
    """
    Region Based Pooling when pooling module operates on multiple channel/region split at once (when the pooling
    module used inherits from MultiChannelSplitPoolingBase)

    todo: As exemplified below, there is nothing which currently forces pooling methods and split methods to have the
     same number of regions (nor is it a check, because not all pooling methods require that input)

    todo: Why multiple pooling methods? they run in a for-loop anyway

    Examples
    --------
    >>> my_p_methods = ("MultiCSSharedRocket", "MultiCSSharedRocket")
    >>> my_p_kwargs = ({"num_regions": (3, 7, 4), "num_kernels": 43, "max_receptive_field": 37},
    ...                {"num_regions": (11, 4, 9, 8), "num_kernels": 57, "max_receptive_field": 43})
    >>> my_s_methods = (("VoronoiSplit", "VoronoiSplit", "VoronoiSplit"),
    ...                 ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit", "VoronoiSplit"))
    >>> my_box = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
    >>> my_s_kwargs = (({"num_points": 3, **my_box}, {"num_points": 7, **my_box}, {"num_points": 4, **my_box}),
    ...                ({"num_points": 11, **my_box}, {"num_points": 4, **my_box}, {"num_points": 9, **my_box},
    ...                 {"num_points": 8, **my_box}))
    >>> my_model = __DeprecatedMultiChannelSplitsRegionBasedPooling(pooling_methods=my_p_methods,
    ...                                                 pooling_methods_kwargs=my_p_kwargs, split_methods=my_s_methods,
    ...                                                 split_methods_kwargs=my_s_kwargs)
    >>> my_model.supports_precomputing
    True
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
        split_methods : tuple[tuple[str, ...], ...]
            Region splits method. Must have the same length as pooling_methods. The length of each element must further
            be equal to the number of regions in the corresponding channel/region split
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
        if not all(isinstance(pooling_module, MultiChannelSplitsPoolingBase) for pooling_module in pooling_modules):
            wrong_methods = tuple(pooling_method for pooling_method in pooling_methods
                                  if not isinstance(pooling_method, MultiChannelSplitsPoolingBase))
            raise TypeError(f"Expected all pooling methods to inherit from {MultiChannelSplitsPoolingBase.__name__}, "
                            f"but found {wrong_methods}")

        # Pass them to nn.ModuleList (otherwise they are not registered as modules with parameters py pytorch)
        self._pooling_modules = nn.ModuleList(pooling_modules)

        # -------------------
        # Channel/Region splits
        # -------------------
        # Generate and store region splits
        region_splits: List[Tuple[RegionSplitBase, ...]] = []
        for s_methods, s_kwargs in zip(split_methods, split_methods_kwargs):
            region_splits.append(
                tuple(get_region_split(method, **kwargs) for method, kwargs in zip(s_methods, s_kwargs))
            )
        self._region_splits = tuple(region_splits)

        # Initialise the mapping from regions to channel names, for all datasets (must be fit later)
        # Should be {dataset_name: tuple[tuple[ChannelsInRegionSplit, ...], ...]}
        self._channel_splits: Dict[str, Tuple[Tuple[ChannelsInRegionSplit, ...], ...]] = dict()

    # ----------------
    # Checks
    # ----------------
    @staticmethod
    def _input_checks(pooling_methods, pooling_methods_kwargs, split_methods, split_methods_kwargs):
        # Length check
        expected_same_lengths = (pooling_methods, pooling_methods_kwargs, split_methods, split_methods_kwargs)
        if not all(len(arg) == len(pooling_methods) for arg in expected_same_lengths):
            raise ValueError(f"Expected pooling methods, their corresponding kwargs, and the region splits methods and "
                             f"their corresponding kwargs all to have the same lengths, but found lengths "
                             f"{(tuple(len(arg) for arg in expected_same_lengths))}")

    @staticmethod
    def _check_num_channel_splits_matching(pooling_modules, split_methods, split_methods_kwargs):
        for pooling_module, s_method, s_kwargs in zip(pooling_modules, split_methods, split_methods_kwargs):
            if not (pooling_module.num_channel_splits == len(s_method) == len(s_kwargs)):
                raise ValueError(f"Expected number of channel/region splits of the pooling module to match that of the "
                                 f"actual channel/region split and corresponding kwargs, but found "
                                 f"{pooling_module.num_channel_splits}, {len(s_method)}, and {len(s_kwargs)}")

    # ----------------
    # Forward and pre-computing
    # ----------------
    def forward(self, x, *, channel_system_name, channel_name_to_index, pre_computed=None):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        channel_system_name : str
        channel_name_to_index : dict[str, int]
        pre_computed : tuple, optional

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        # ------------------
        # Pass through all channel splits and return as tuple
        # ------------------
        # Simple case when no pre-computing is made
        if not self.supports_precomputing or pre_computed is None:
            # Compute outputs
            outputs = tuple(pooling_module(x, channel_splits=channel_splits,
                                           channel_name_to_index=channel_name_to_index)
                            for pooling_module, channel_splits in zip(self._pooling_modules,
                                                                      self._channel_splits[channel_system_name]))
            # Unpack and return
            return tuple(itertools.chain(*outputs))

        # Otherwise, append to a list
        output_channel_splits: List[Tuple[torch.Tensor, ...]] = []
        for pre_comp_features, pooling_module, channel_splits \
                in zip(pre_computed, self._pooling_modules, self._channel_splits[channel_system_name]):
            # Handle the unsupported case, or when pre-computing is not desired
            if not pooling_module.supports_precomputing() or pre_comp_features is None:
                output_channel_splits.append(pooling_module(x, channel_splits=channel_splits,
                                                            channel_name_to_index=channel_name_to_index))
            else:
                output_channel_splits.append(pooling_module(x, channel_splits=channel_splits,
                                                            channel_name_to_index=channel_name_to_index,
                                                            pre_computed=pre_comp_features))
        # Convert to unpacked tuple and return
        return tuple(itertools.chain(*output_channel_splits))

    def pre_compute(self, x):
        """
        Method for pre-computing

        todo: copied from SingleChannelSplitRegionBasedPooling
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of pre-computed tensor (one pr. pooling module). The element will be None if the corresponding
            pooling module does not support pre-computing
        """
        # Quick check to see if this method should be run
        if not self.supports_precomputing:
            raise RuntimeError("Tried to pre-compute when no pooling method supports pre-computing")

        # Loop through all pooling modules
        pre_computed: List[Optional[torch.Tensor]] = []
        for pooling_module in self._pooling_modules:
            if pooling_module.supports_precomputing():
                # Assuming that the method is called 'pre_compute', and that it only takes in 'x' as argument
                pre_computed.append(pooling_module.pre_compute(x))
            else:
                pre_computed.append(None)

        # Convert to tuple and return
        return tuple(pre_computed)

    # ----------------
    # Methods for fitting channel systems
    # ----------------
    def fit_channel_system(self, channel_system):
        """
        Fit a single channel system on the regions splits

        Parameters
        ----------
        channel_system : cdl_eeg.data.datasets.dataset_base.ChannelSystem
            The channel system to fit
        Returns
        -------
        None
        """
        # Loop through all multi-region splits
        multi_channel_splits = []
        for region_splits in self._region_splits:
            # Loop through all region splits in a single multi-region split
            channel_splits = []
            for region_split in region_splits:
                # Place the electrodes
                channel_splits.append(region_split.place_in_regions(channel_system.electrode_positions))
            multi_channel_splits.append(tuple(channel_splits))

        self._channel_splits[channel_system.name] = tuple(multi_channel_splits)

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
            raise TypeError(
                f"Expected channel systems to be either a channel system (type={ChannelSystem.__name__}) "
                f"or a tuple of channel systems, but this was not the case")

    # ----------------
    # Properties
    # ----------------
    @property
    def channel_splits(self):
        return self._channel_splits

    @property
    def supports_precomputing(self):
        # todo: move to base class
        return any(pooling_module.supports_precomputing() for pooling_module in self._pooling_modules)


# ------------------
# 'The' RBP implementation
# ------------------
class RegionBasedPooling(nn.Module):
    """
    The main implementation of Region Based Pooling (Tveitstøl et al., 2023, submitted)
    """

    def __init__(self, rbp_designs):
        """
        Initialise

        Parameters
        ----------
        rbp_designs : tuple[RBPDesign, ...]
        """
        super().__init__()

        # ------------------
        # Create all RBP modules
        # ------------------
        rbp_modules: List[RegionBasedPoolingBase] = []
        for design in rbp_designs:
            design: RBPDesign

            # Select the correct class
            if design.pooling_type == RBPPoolType.SINGLE_CS:
                rbp = SingleChannelSplitRegionBasedPooling(pooling_methods=design.pooling_methods,
                                                           pooling_methods_kwargs=design.pooling_methods_kwargs,
                                                           split_methods=design.split_methods,
                                                           split_methods_kwargs=design.pooling_methods_kwargs)
            elif design.pooling_type == RBPPoolType.MULTI_CS:
                rbp = MultiChannelSplitsRegionBasedPooling(pooling_method=design.pooling_methods,
                                                           pooling_method_kwargs=design.pooling_methods_kwargs,
                                                           split_methods=design.split_methods,
                                                           split_methods_kwargs=design.pooling_methods_kwargs)
            else:
                raise ValueError(f"Expected pooling type to be in {tuple(type_ for type_ in RBPPoolType)}, but found "
                                 f"{design.pooling_type}")

            # Append the object to rbp_modules
            rbp_modules.append(rbp)

        # Store all in a ModuleList to register the modules properly
        self._rbp_modules = nn.ModuleList(rbp_modules)
