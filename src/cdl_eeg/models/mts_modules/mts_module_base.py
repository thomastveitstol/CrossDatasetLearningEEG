import abc
from typing import Any, Dict

import torch.nn as nn


class MTSModuleBase(nn.Module, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def sample_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
        """Method for sampling hyperparameters from a config file containing distributions of which to sample from"""

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
    def successful_initialisation(cls, *args, **kwargs):
        """Method which returns True if the provided hyperparameters will give a successful initialisation, False if a
        ValueError or ZeroDivisionError is raised. This was implemented as the braindecode models are not always able to
        handle the input dimensionality, and tend to raise a ValueError or ZeroDivisionError if the input time series is
        too short for the architecture to handle"""
        try:
            cls(*args, **kwargs)  # type: ignore[call-arg]
        except (ValueError, ZeroDivisionError):
            return False
        return True

    @classmethod
    def get_latent_features_dim(cls, *args, **kwargs):
        """Get the expected latent feature dimension"""
        return cls(*args, **kwargs).latent_features_dim  # type: ignore[call-arg]

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self) -> int:
        raise NotImplementedError
