import abc
from typing import List

import torch.nn as nn


class MTSModuleBase(nn.Module, abc.ABC):

    requires_input_time_steps = False  # Set this to true if the class requires knowing number of input time steps

    @classmethod
    def supports_latent_feature_extraction(cls):
        """
        Check if the class supports pre-computing by checking if there are any methods with the
        'latent_feature_extraction_method' decorator

        Returns
        -------
        bool
            True if the inheriting module supports latent feature extraction, False otherwise
        """
        # Get all methods
        methods = tuple(getattr(cls, method) for method in dir(cls) if callable(getattr(cls, method)))

        # Check if any of the methods are decorated as a latent_feature_extraction_method
        return any(getattr(method, "_is_latent_feature_extraction_method", False) for method in methods)

    @classmethod
    def get_available_latent_feature_extractions(cls):
        """
        Get all latent feature extraction methods available for the class. The target method must be decorated by
        @latent_feature_extraction_method to be properly registered

        Returns
        -------
        tuple[str, ...]
        """
        # Get all target methods
        feature_extraction_methods: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a latent feature extraction method
            if callable(attribute) and getattr(attribute, "_is_latent_feature_extraction_method", False):
                feature_extraction_methods.append(method)

        # Convert to tuple and return
        return tuple(feature_extraction_methods)

    def extract_latent_features(self, input_tensor):
        """
        Method for extracting latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
            Latent features
        """
        raise NotImplementedError

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features extracted

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        raise NotImplementedError

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self):
        raise NotImplementedError
