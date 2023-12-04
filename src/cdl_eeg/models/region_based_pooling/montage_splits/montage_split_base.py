import abc


class MontageSplitBase(abc.ABC):
    """
    Base class for splitting the EEG cap into regions
    """

    __slots__ = ()

    @abc.abstractmethod
    def place_in_regions(self, electrodes_3d):
        """
        Method for placing multiple electrodes into regions

        Parameters
        ----------
        electrodes_3d : cdl_eeg.models.region_based_pooling.utils.Electrodes3D

        Returns
        -------
        cdl_eeg.models.region_based_pooling.utils.ChannelsInRegionSplit
        """

    @abc.abstractmethod
    def plot(self, **kwargs):
        """
        Method for plotting the regions. Although not mathematically crucial, implementing plotting is important for
        both debugging and visualisation (needed to explain other people)

        Returns
        -------
        None
        """
