import abc

import torch.nn as nn


class PoolingModuleBase(nn.Module, abc.ABC):  # todo: not sure if inheriting from abc.ABC is important pr. now
    """
    Pooling module base
    """


class SingleRegionPoolingBase(PoolingModuleBase):
    """
    Base class for pooling modules operating on a single region. This offers great flexibility, but may come at a
    computational cost (for-loops are slow)
    """


class MultiRegionPoolingBase(PoolingModuleBase):
    """
    Base class for pooling modules operating on a subset of regions in a single channel split. If the idea is to operate
    on all regions in a region/channel split, inherit from SingleChannelSplitPoolingBase instead
    """


class SingleChannelSplitPoolingBase(PoolingModuleBase):
    """
    Base class for pooling modules operating on an entire channel split. This may be preferred to single regions, as it
    allows for more time-efficient implementations
    """


class MultiChannelSplitsPoolingBase(PoolingModuleBase):
    """
    Base class for pooling modules operating on multiple channel splits. This may be preferred, as it allows for more
    time-efficient implementations
    """
