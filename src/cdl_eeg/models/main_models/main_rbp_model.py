import torch
import torch.nn as nn

from cdl_eeg.models.mts_modules.getter import get_mts_module
from cdl_eeg.models.region_based_pooling.region_based_pooling import SingleChannelSplitRegionBasedPooling


class MainSingleChannelSplitRBPModel(nn.Module):
    """
    (In early stages of development)

    Main model when using RBP (SingleChannelSplitRegionBasedPooling class) as first layer

    PS: Merges channel splits by concatenation

    todo: should have a MainModel base class

    Examples
    --------
    >>> my_split_kwargs = ({"num_points": 7, "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
    ...                    {"num_points": 11, "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
    >>> _ = MainSingleChannelSplitRBPModel(mts_module="InceptionTime",
    ...                                    mts_module_kwargs={"in_channels": 8, "num_classes": 5},
    ...                                    pooling_methods=("SingleCSMean", "SingleCSMean"),
    ...                                    pooling_methods_kwargs=({}, {}),
    ...                                    split_methods=("VoronoiSplit", "VoronoiSplit"),
    ...                                    split_methods_kwargs=my_split_kwargs)
    """

    def __init__(self, *, mts_module, mts_module_kwargs, pooling_methods, pooling_methods_kwargs, split_methods,
                 split_methods_kwargs):
        super().__init__()

        # -----------------
        # Create RBP layer
        # -----------------
        self._region_based_pooling = SingleChannelSplitRegionBasedPooling(pooling_methods=pooling_methods,
                                                                          pooling_methods_kwargs=pooling_methods_kwargs,
                                                                          split_methods=split_methods,
                                                                          split_methods_kwargs=split_methods_kwargs)

        # ----------------
        # Create MTS module
        # ----------------
        # todo: the number of in channels must be calculated from/by the RBP layer
        self._mts_module = get_mts_module(mts_module_name=mts_module, **mts_module_kwargs)

    def forward(self, x, *, channel_system_name, channel_name_to_index):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        channel_system_name : str
        channel_name_to_index : dict[str, int]

        Returns
        -------
        torch.Tensor
            Outputs of MTS module without applying a final activation function
        """
        # TODO: Would be nice to be able to have different channel systems in the same batch
        # Pass through RBP layer
        x = self._region_based_pooling(x, channel_system_name=channel_system_name,
                                       channel_name_to_index=channel_name_to_index)

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
