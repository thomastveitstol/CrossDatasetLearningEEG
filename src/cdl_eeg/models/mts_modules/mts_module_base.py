import abc
from typing import List, Any, Dict

import torch.nn as nn


class MTSModuleBase(nn.Module, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def sample_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
        """Method for sampling hyperparameters from a config file containing distributions of which to sample from"""

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

    @classmethod
    def successful_initialisation(cls, *args, **kwargs):  # type: ignore[call-arg]
        """Method which returns True if the provided hyperparameters will give a successful initialisation, False if a
        ValueError or ZeroDivisionError is raised. This was implemented as the braindecode models raises are not always
        able to handle the input dimensionality, and tends to raise a ValueError or ZeroDivisionError if the input time
        series is too short for the architecture to handle"""
        try:
            cls(*args, **kwargs)
        except (ValueError, ZeroDivisionError):  # todo: consider raising some other error, a more specific one
            return False
        return True

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self):
        raise NotImplementedError
