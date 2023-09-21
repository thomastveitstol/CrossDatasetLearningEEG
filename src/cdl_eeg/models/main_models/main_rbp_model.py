import torch
import torch.nn as nn

from cdl_eeg.models.mts_modules.getter import get_mts_module
from cdl_eeg.models.region_based_pooling.region_based_pooling import RegionBasedPooling


class MainRBPModel(nn.Module):
    """
    (In early stages of development)

    Main model supporting use of RBP. That is, this class uses RBP as a first layer, followed by an MTS module

    PS: Merges channel splits by concatenation
    """

    def __init__(self, *, mts_module, mts_module_kwargs, rbp_designs):
        """
        Initialise

        Parameters
        ----------
        mts_module : str
        mts_module_kwargs : dict[str, typing.Any]
        rbp_designs : tuple[cdl_eeg.models.region_based_pooling.region_based_pooling.RBPDesign, ...]
        """
        super().__init__()

        # -----------------
        # Create RBP layer
        # -----------------
        self._region_based_pooling = RegionBasedPooling(rbp_designs)

        # ----------------
        # Create MTS module
        # ----------------
        # todo: the number of in channels must be calculated from/by the RBP layer
        self._mts_module = get_mts_module(mts_module_name=mts_module, **mts_module_kwargs)

    def pre_compute(self, x):
        """
        Pre-compute

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        tuple[torch.Tensor, ...]
        """
        return self._region_based_pooling.pre_compute(x)

    def forward(self, x, *, channel_system_name, channel_name_to_index, pre_computed=None):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        channel_system_name : str
        channel_name_to_index : dict[str, int]
        pre_computed : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
            Outputs of MTS module without applying a final activation function
        """
        # TODO: Would be nice to be able to have different channel systems in the same batch
        # Pass through RBP layer
        x = self._region_based_pooling(x, channel_system_name=channel_system_name,
                                       channel_name_to_index=channel_name_to_index, pre_computed=pre_computed)

        # Merge by concatenation
        x = torch.cat(x, dim=1)

        # Pass through MTS module and return
        return self._mts_module(x)

    # ----------------
    # Methods for fitting channel systems
    # todo: these should probably be moved to a base class
    # ----------------
    def fit_channel_system(self, channel_system):
        self._region_based_pooling.fit_channel_system(channel_system)

    def fit_channel_systems(self, channel_systems):
        self._region_based_pooling.fit_channel_systems(channel_systems)
