import abc

import torch.nn as nn


class SingleRegionPoolingBase(nn.Module, abc.ABC):
    """
    Base class for pooling modules operating on a single region. This offers great flexibility, but typically comes at a
    computational cost (for-loops are slow)
    """


class MultiRegionPoolingBase(nn.Module, abc.ABC):
    """
    Base class for pooling modules operating on a subset of regions in a single channel split.
    """


class SingleChannelSplitPoolingBase(nn.Module, abc.ABC):
    """
    Base class for pooling modules operating on an entire channel split. This may be preferred to single regions, as it
    allows for more time-efficient implementations
    """


class MultiChannelSplitsPoolingBase(nn.Module, abc.ABC):
    """
    Base class for pooling modules operating on multiple channel splits. This may be preferred, as it allows for more
    time-efficient implementations
    """
