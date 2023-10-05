import abc
from typing import List

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

    @classmethod
    def get_available_latent_feature_extractions(cls):
        """Get all latent feature extraction methods available for the class. The target method must be decorated by
        @latent_feature_extraction_method to be properly registered"""
        # Get all target methods
        feature_extraction_methods: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a latent feature extraction method
            if callable(attribute) and getattr(attribute, "_is_latent_feature_extraction_method", False):
                feature_extraction_methods.append(method)

        # Convert to tuple and return
        return tuple(feature_extraction_methods)

    def extract_latent_features(self, data, method="default_latent_feature_extraction"):
        # Input check
        if method not in self.get_available_latent_feature_extractions():
            raise ValueError(f"Latent feature extraction method '{method}' was not recognised. Make sure that the "
                             f"method passed shares the name with the implemented method you want to use. The latent "
                             f"feature extraction methods available for this class ({type(self).__name__}) are: "
                             f"{self.get_available_latent_feature_extractions()}")

        # Run and return
        return getattr(self, method)(data)
