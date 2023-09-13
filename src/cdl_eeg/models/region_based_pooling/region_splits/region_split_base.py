import abc


class RegionSplitBase(abc.ABC):
    """
    Base class for splitting the EEG cap into regions
    """

    __slots__ = ()

    @abc.abstractmethod
    def place_in_regions(self, electrode_positions):
        """
        Method for placing multiple electrodes into regions

        Parameters
        ----------
        electrode_positions : tuple[cdl_eeg.models.region_based_pooling.utils.CartCoord, ...]

        Returns
        -------
        cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit
        """
