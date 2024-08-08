"""
Classes for Region Based Pooling

There will likely be some overlap with a former project (where Region Based Pooling was first introduced) at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/models/modules/bins/regions_to_bins.py

Note that 'channel split' and 'region split' are occasionally wrongly used instead of 'montage split'. Some of the code
was made before I settled on the term 'montage split'.

Authored by:
    Thomas Tveitstøl (Oslo University Hospital)
"""
import abc
import dataclasses
import itertools
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cdl_eeg.data.datasets.dataset_base import ChannelSystem
from cdl_eeg.models.domain_adaptation.cmmn import RBPConvMMN
from cdl_eeg.models.region_based_pooling.pooling_modules.getter import get_pooling_module
from cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base import MultiMontageSplitsPoolingBase
from cdl_eeg.models.region_based_pooling.montage_splits.getter import get_montage_split
from cdl_eeg.models.region_based_pooling.utils import CHANNELS_IN_MONTAGE_SPLIT


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
    pooling_type: RBPPoolType
    pooling_methods: Union[str, Tuple[str, ...]]
    pooling_methods_kwargs: Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]
    split_methods: Union[str, Tuple[str, ...]]
    split_methods_kwargs: Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]
    use_cmmn_layer: bool = False
    cmmn_kwargs: Optional[Dict[str, Any]] = None
    num_designs: int = 1


# ------------------
# Base class
# ------------------
class RegionBasedPoolingBase(nn.Module, abc.ABC):
    """
    Base class for all Region Based Pooling classes
    """

    @property
    @abc.abstractmethod
    def num_regions(self) -> int:
        """
        Get the total number of regions

        Returns
        -------
        int
            Total number of regions
        """


# ------------------
# Implementations of RBP
# ------------------
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
    >>> my_split_kwargs = ({"min_nodes": 1, "channel_systems": ("HatlestadHall",), **my_box},
    ...                    {"min_nodes": 1, "channel_systems": ("HatlestadHall",), **my_box},
    ...                    {"min_nodes": 1, "channel_systems": ("HatlestadHall",), **my_box})
    >>> my_model = MultiChannelSplitsRegionBasedPooling(pooling_method=my_pooling_method,
    ...                                                 pooling_method_kwargs=my_pooling_kwargs,
    ...                                                 split_methods=my_split_methods,
    ...                                                 split_methods_kwargs=my_split_kwargs,
    ...                                                 use_cmmn_layer=True,
    ...                                                 cmmn_kwargs={"kernel_size": 32})
    >>> my_model.supports_precomputing
    True
    """

    def __init__(self, pooling_method, pooling_method_kwargs, split_methods, split_methods_kwargs, use_cmmn_layer,
                 cmmn_kwargs):
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
        # Montage splits
        # -------------------
        # Generate and store region/montage splits
        self._region_splits = tuple(get_montage_split(split_method, **kwargs)
                                    for split_method, kwargs in zip(split_methods, split_methods_kwargs))

        # Initialise the mapping from regions to channel names, for all datasets (must be fit later)
        # Should be {dataset_name: tuple[CHANNELS_IN_MONTAGE_SPLIT, ...]}
        self._channel_splits: Dict[str, Tuple[CHANNELS_IN_MONTAGE_SPLIT, ...]] = dict()

        # -------------------
        # (Maybe) use RBP compatible CMMN layer
        # -------------------
        self._cmmn_layer = None if not use_cmmn_layer \
            else RBPConvMMN(**{"num_montage_splits": len(self._region_splits), **cmmn_kwargs})

        # -------------------
        # Generate pooling modules
        # -------------------
        # Get correct pooling module in a list
        _num_regions = tuple(split.num_regions for split in self._region_splits)
        pooling_module = get_pooling_module(pooling_method, **{"num_regions": _num_regions, **pooling_method_kwargs})

        # Verify that it has have correct type
        if not isinstance(pooling_module, MultiMontageSplitsPoolingBase):
            raise TypeError(f"Expected all pooling module to inherit from {MultiMontageSplitsPoolingBase.__name__}, "
                            f"but found {type(pooling_module)}")

        # Store pooling module
        self._pooling_module = pooling_module

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
    def forward(self, input_tensors, *, channel_name_to_index, pre_computed=None):
        """
        Forward method

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
        channel_name_to_index : dict[str, dict[str, int]]
        pre_computed : dict[str, tuple], optional

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        # ------------------
        # Pass through all channel splits
        # ------------------
        # Pass pre_computed or not, depending on compatibility and if pre-computing is not desired (value is None)
        if not self.supports_precomputing or pre_computed is None:
            region_representations = self._pooling_module(input_tensors, channel_splits=self._channel_splits,
                                                          channel_name_to_index=channel_name_to_index)
        else:
            region_representations = self._pooling_module(input_tensors, channel_splits=self._channel_splits,
                                                          channel_name_to_index=channel_name_to_index,
                                                          pre_computed=pre_computed)

        # ------------------
        # Maybe apply CMMN
        # ------------------
        if self.has_cmmn_layer:
            _sizes = 0
            dataset_indices = dict()
            for name, tensor in input_tensors.items():
                dataset_indices[name] = tuple(range(_sizes, _sizes+tensor.size()[0]))
                _sizes += tensor.size()[0]

            region_representations = self._cmmn_layer(region_representations,  # type: ignore[misc]
                                                      dataset_indices=dataset_indices)

        return region_representations

    def pre_compute(self, input_tensors):
        """
        Method for pre-computing

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]

        Returns
        -------
        dict[str, torch.Tensor]
            Pre-computed tensors  todo: is this correct?
        """
        # Quick check to see if this method should be run
        if not self.supports_precomputing:
            raise RuntimeError("Tried to pre-compute when no pooling method supports pre-computing")

        # todo: Assuming that the method is called 'pre_compute', and that it only takes in 'x' as argument
        # mypy thinks I am calling on a Tensor here... why?
        return self._pooling_module.pre_compute(input_tensors)  # type: ignore[operator]

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
    # Methods for fitting CMMN layer
    # ----------------
    def fit_psd_barycenters(self, data, *, channel_systems: Dict[str, ChannelSystem], sampling_freq=None):
        if self._cmmn_layer is None:
            raise RuntimeError("Cannot fit PSD barycenters of the CMMN layers, when none is used")

        # -----------------
        # Update the channel splits of the CMMN layer to what it is in this layer
        # -----------------
        self._cmmn_layer.update_channel_splits(self._channel_splits)

        # -----------------
        # Fit PSD barycenters
        # -----------------
        # Check dimensions
        _sizes = set(d.shape for d in data.values())
        assert all(len(_size) in (3, 4) for _size in _sizes), (
            f"Expected all input data values to be 3D or 4D with shape=(subjects, channels, time_steps), or "
            f"(subjects, eeg_epochs, channels, time_steps) but found: {_sizes}"
        )

        self._cmmn_layer.fit_psd_barycenters(data, channel_systems=channel_systems, sampling_freq=sampling_freq)

    def fit_monge_filters(self, data, *, channel_systems: Dict[str, ChannelSystem]):
        if self._cmmn_layer is None:
            raise RuntimeError("Cannot fit monge filters of the CMMN layers, when none is used")

        # Update channel splits to what it is in this layer
        self._cmmn_layer.update_channel_splits(self._channel_splits)

        # Fit monge filters
        self._cmmn_layer.fit_monge_filters(data, channel_systems=channel_systems)

    # ----------------
    # Properties
    # ----------------
    @property
    def channel_splits(self):
        return self._channel_splits

    @property
    def supports_precomputing(self):
        return self._pooling_module.supports_precomputing()

    @property
    def num_regions(self) -> int:
        return sum(split.num_regions for split in self._region_splits)

    @property
    def has_cmmn_layer(self) -> bool:
        return self._cmmn_layer is not None

    @property
    def cmmn_fitted_channel_systems(self):
        """Get the channel systems which has already been fit"""
        if self.has_cmmn_layer:
            return self._cmmn_layer.fitted_channel_systems  # type: ignore[union-attr]
        else:
            return ()


# ------------------
# 'The' RBP implementation
# ------------------
class RegionBasedPooling(nn.Module):
    """
    The main implementation of Region Based Pooling (Tveitstøl et al., 2023, submitted)

    Examples
    --------
    >>> p_method_0 = "MultiCSSharedRocket"
    >>> p_kwargs_0 = {"num_regions": (1, 1, 1), "num_kernels": 43, "max_receptive_field": 37}
    >>> s_methods_0 = ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit")
    >>> my_box = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
    >>> s_kwargs_0 = ({"min_nodes": 1, "channel_systems": ("HatlestadHall",),  **my_box},
    ...               {"min_nodes": 1, "channel_systems": ("HatlestadHall",), **my_box},
    ...               {"min_nodes": 1, "channel_systems": ("HatlestadHall",), **my_box})
    >>> design_0 = RBPDesign(pooling_type=RBPPoolType.MULTI_CS, pooling_methods=p_method_0,
    ...                      pooling_methods_kwargs=p_kwargs_0, split_methods=s_methods_0,
    ...                      split_methods_kwargs=s_kwargs_0, num_designs=2)
    >>> p_method_1 = "MultiCSSharedRocket"
    >>> p_kwargs_1 = {"num_regions": (1, 1, 1), "num_kernels": 32, "max_receptive_field": 11}
    >>> s_methods_1 = ("VoronoiSplit", "VoronoiSplit", "VoronoiSplit")
    >>> s_kwargs_1 = ({"min_nodes": 1, "channel_systems": ("HatlestadHall",),  **my_box},
    ...               {"min_nodes": 1, "channel_systems": ("HatlestadHall",), **my_box},
    ...               {"min_nodes": 1, "channel_systems": ("HatlestadHall",), **my_box})
    >>> design_1 = RBPDesign(pooling_type=RBPPoolType.MULTI_CS, pooling_methods=p_method_1,
    ...                      pooling_methods_kwargs=p_kwargs_1, split_methods=s_methods_1,
    ...                      split_methods_kwargs=s_kwargs_1, num_designs=3)
    >>> _ = RegionBasedPooling((design_0, design_1))
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

            # Create as many similar designs as specified
            for _ in range(design.num_designs):
                # Select the correct class
                rbp: RegionBasedPoolingBase
                if design.pooling_type == RBPPoolType.MULTI_CS:
                    rbp = MultiChannelSplitsRegionBasedPooling(
                        pooling_method=design.pooling_methods,
                        pooling_method_kwargs=design.pooling_methods_kwargs,
                        split_methods=design.split_methods,
                        split_methods_kwargs=design.split_methods_kwargs,
                        use_cmmn_layer=design.use_cmmn_layer,
                        cmmn_kwargs={} if design.cmmn_kwargs is None else design.cmmn_kwargs
                    )
                else:
                    raise ValueError(f"Expected pooling type to be in {tuple(type_ for type_ in RBPPoolType)}, but "
                                     f"found {design.pooling_type}")

                # Append the object to rbp_modules
                rbp_modules.append(rbp)

        # Store all in a ModuleList to register the modules properly
        self._rbp_modules = nn.ModuleList(rbp_modules)

    # ----------------
    # Forward and pre-computing
    # ----------------
    def forward(self, input_tensors, *, channel_name_to_index, pre_computed=None):
        """
        Forward method

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]
        channel_name_to_index : dict[str, int]
        pre_computed : tuple[dict[str, typing.Any], ...], optional

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        # ------------------
        # Pass through all RBP modules and return as unpacked tuple
        # ------------------
        # Simple case when no pre-computing is made
        if not self.supports_precomputing or pre_computed is None:
            # Compute outputs
            outputs = tuple(
                rbp_module(input_tensors, channel_name_to_index=channel_name_to_index)
                for rbp_module in self._rbp_modules
            )
            # Unpack and return
            return tuple(itertools.chain(*outputs))

        # Otherwise, append to a list
        rbp_outputs: List[Tuple[torch.Tensor, ...]] = []
        for pre_comp_features, rbp_module in zip(pre_computed, self._rbp_modules):
            # Handle the unsupported case, or when pre-computing is not desired
            if not rbp_module.supports_precomputing or pre_comp_features is None:
                # TODO: somewhat hard-coded
                if isinstance(rbp_module, MultiChannelSplitsRegionBasedPooling):
                    rbp_outputs.extend(rbp_module(input_tensors, channel_name_to_index=channel_name_to_index))
                else:
                    rbp_outputs.append(rbp_module(input_tensors, channel_name_to_index=channel_name_to_index))
            else:
                if isinstance(rbp_module, MultiChannelSplitsRegionBasedPooling):
                    rbp_outputs.extend(rbp_module(input_tensors, channel_name_to_index=channel_name_to_index,
                                                  pre_computed=pre_comp_features))
                else:
                    rbp_outputs.append(rbp_module(input_tensors, channel_name_to_index=channel_name_to_index,
                                                  pre_computed=pre_comp_features))
        # Convert to tuple and return
        return tuple(rbp_outputs)

    def pre_compute(self, input_tensors):
        """
        Method for pre-computing

        Parameters
        ----------
        input_tensors : dict[str, torch.Tensor]

        Returns
        -------
        tuple[dict[str, torch.Tensor] | None, ...]
            A tuple of pre-computed tensors (one pr. RBP module). The element will be None if the corresponding
            pooling module does not support pre-computing
        """
        # Quick check to see if this method should be run
        if not self.supports_precomputing:
            raise RuntimeError("Tried to pre-compute when no pooling method supports pre-computing")

        # Loop through all RBP modules
        pre_computed: List[Optional[torch.Tensor]] = []
        for rbp_module in self._rbp_modules:
            if rbp_module.supports_precomputing:
                # Assuming that the method is called 'pre_compute', and that it only takes in 'x' as argument
                pre_computed.append(rbp_module.pre_compute(input_tensors))
            else:
                pre_computed.append(None)

        # Convert to tuple and return
        return tuple(pre_computed)

    # ----------------
    # Methods for fitting channel systems
    # ----------------
    def fit_channel_system(self, channel_system):
        """
        Fit a single channel system on all RBP modules

        Parameters
        ----------
        channel_system : cdl_eeg.data.datasets.dataset_base.ChannelSystem
            The channel system to fit

        Returns
        -------
        None
        """
        for rbp_module in self._rbp_modules:
            rbp_module.fit_channel_system(channel_system)

    def fit_channel_systems(self, channel_systems):
        """
        Fit multiple channel systems on all RBP modules

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
                f"or a tuple of channel systems, but found type {type(channel_systems)}")

    # ----------------
    # Methods for fitting CMMN layer
    # ----------------
    def fit_psd_barycenters(self, data, *, channel_systems: Dict[str, ChannelSystem], sampling_freq=None):
        # If there are no CMMN layers, raise a warning
        if not self.any_cmmn_layers:
            warnings.warn("Trying to fit PSD barycenters of CMMN layers, but none of the RBP modules has one",
                          RuntimeWarning)

        # Loop through all RBP models
        for rbp_module in self._rbp_modules:
            # Only fit the ones which uses CMMN layer
            if rbp_module.has_cmmn_layer:
                rbp_module.fit_psd_barycenters(data=data, channel_systems=channel_systems, sampling_freq=sampling_freq)

    def fit_monge_filters(self, data, *, channel_systems: Dict[str, ChannelSystem]):
        # If there are no CMMN layers, raise a warning
        if not self.any_cmmn_layers:
            warnings.warn("Trying to fit monge filters of CMMN layers, but none of the RBP modules has one",
                          RuntimeWarning)

        # Loop through all RBP models
        for rbp_module in self._rbp_modules:
            # Only fit the ones which uses CMMN layer
            if rbp_module.has_cmmn_layer:
                rbp_module.fit_monge_filters(data=data, channel_systems=channel_systems)

    # ----------------
    # Properties
    # ----------------
    @property
    def num_regions(self) -> int:
        """Get the total number of regions"""
        return sum(rbp.num_regions for rbp in self._rbp_modules)

    @property
    def supports_precomputing(self):
        return any(rbp_module.supports_precomputing for rbp_module in self._rbp_modules)

    @property
    def any_cmmn_layers(self) -> bool:
        """Boolean which indicates if any of the RBP modules has a CMMN layer (True) or not (False)"""
        return any(rbp_module.has_cmmn_layer for rbp_module in self._rbp_modules)

    @property
    def cmmn_fitted_channel_systems(self):
        """Get the channel systems which has already been fit"""
        # Loop through all RBP models
        fitted_channel_systems: Tuple[str, ...] = tuple()
        for rbp_module in self._rbp_modules:
            # Check only the ones which has CMMN layer
            if rbp_module.has_cmmn_layer:
                # All RBP modules should have the same channel systems fitted
                if fitted_channel_systems:
                    if fitted_channel_systems != rbp_module.cmmn_fitted_channel_systems:
                        raise RuntimeError("Expected all RBP modules with CMMN layer to be fit on the same channel "
                                           "systems, but this was not the case")
                else:
                    fitted_channel_systems = rbp_module.cmmn_fitted_channel_systems
        return fitted_channel_systems
