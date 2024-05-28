import abc

import torch.nn as nn


# --------------------
# Convenient decorators
# --------------------
def precomputing_method(func):
    setattr(func, "_is_precomputing_method", True)
    return func


# --------------------
# Base classes
# --------------------
class PoolingModuleBase(nn.Module, abc.ABC):
    """
    Pooling module base
    """

    @classmethod
    def supports_precomputing(cls):
        """Check if the class supports pre-computing by checking if there are any methods with the 'precomputing_method'
        decorator"""
        # Get all methods
        methods = tuple(getattr(cls, method) for method in dir(cls) if callable(getattr(cls, method)))

        # Check if any of the methods are decorated as a precomputing_method
        return any(getattr(method, "_is_precomputing_method", False) for method in methods)


class MultiMontageSplitsPoolingBase(PoolingModuleBase):
    """
    Base class for pooling modules operating on multiple montage splits. This may be preferred, as it allows for more
    time-efficient implementations
    """
