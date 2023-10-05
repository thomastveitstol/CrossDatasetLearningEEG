import abc

import torch.nn as nn


def latent_feature_extraction_method(func):
    setattr(func, "_is_latent_feature_extraction_method", True)
    return func


class MTSModuleBase(nn.Module, abc.ABC):
    # todo: consider overriding __new__ to store all inputs and defaults of __init__

    @classmethod
    def supports_latent_feature_extraction(cls):
        """Check if the class supports pre-computing by checking if there are any methods with the
        'latent_feature_extraction_method' decorator"""
        # Get all methods
        methods = tuple(getattr(cls, method) for method in dir(cls) if callable(getattr(cls, method)))

        # Check if any of the methods are decorated as a latent_feature_extraction_method
        return any(getattr(method, "_is_latent_feature_extraction_method", False) for method in methods)

