import abc


class ScalerBase(abc.ABC):

    __slots__ = ()

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        Method for fitting the scaler

        Returns
        -------
        None
        """

    @abc.abstractmethod
    def transform(self, data):
        """
        Method for transforming, based on the parameters and nature of the scaling

        Parameters
        ----------
        data

        Returns
        -------
        Scaled data
        """

    @abc.abstractmethod
    def inv_transform(self, scaled_data):
        """
        Method for re-transforming, based on the parameters and nature of the scaling

        Parameters
        ----------
        scaled_data
            Scaled data

        Returns
        -------
        De-scaled data
        """


class TargetScalerBase(ScalerBase, abc.ABC):
    __slots__ = ()


class InputScalerBase(ScalerBase, abc.ABC):
    __slots__ = ()
