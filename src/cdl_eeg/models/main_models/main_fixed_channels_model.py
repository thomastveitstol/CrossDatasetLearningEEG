import torch
import torch.nn as nn

from cdl_eeg.models.mts_modules.getter import get_mts_module


class MainFixedChannelsModel(nn.Module):
    """
    Main model when the number of input channels is fixed
    """

    def __init__(self, mts_module, **kwargs):
        super().__init__()

        # ----------------
        # Create MTS module
        # ----------------
        self._mts_module = get_mts_module(mts_module_name=mts_module, **kwargs)

    @classmethod
    def from_config(cls, config):
        return cls(mts_module=config["name"], **config["kwargs"])

    def forward(self, x):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]

        Returns
        -------
        torch.Tensor
            Outputs of MTS module without applying a final activation function

        Examples
        --------
        >>> my_model = MainFixedChannelsModel("InceptionTime", in_channels=23, num_classes=11)
        >>> my_model(torch.rand(size=(10, 23, 300))).size()
        torch.Size([10, 11])
        >>> my_model({"d1": torch.rand(size=(10, 23, 300)), "d2": torch.rand(size=(4, 23, 300))}).size()
        torch.Size([14, 11])
        """
        # (Maybe) concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            x = torch.cat(tuple(x.values()), dim=0)

        # Run through MTS module and return
        return self._mts_module(x)

    def extract_latent_features(self, x, method="default_latent_feature_extraction"):
        """Method for extracting latent features"""
        # Input check
        if not self._mts_module.supports_latent_feature_extraction():
            raise ValueError(f"The MTS module {type(self._mts_module).__name__} does not support latent feature "
                             f"extraction")

        # (Maybe) concatenate all tensors. This should be possible, as this class should ony be used with a fixed number
        # of input channels
        if isinstance(x, dict):
            x = torch.cat(tuple(x.values()), dim=0)

        # Run through MTS module and return
        return self._mts_module.extract_latent_features(x, method=method)
