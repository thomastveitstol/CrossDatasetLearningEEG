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
        data : dict[str, numpy.ndarray]

        Returns
        -------
        dict[str, numpy.ndarray]
        """

    @abc.abstractmethod
    def inv_transform(self, scaled_data):
        """
        Method for re-transforming, based on the parameters and nature of the scaling

        Parameters
        ----------
        scaled_data : dict[str, numpy.ndarray]
            Scaled data

        Returns
        -------
        dict[str, numpy.ndarray]
            De-scaled data
        """


class TargetScalerBase(ScalerBase, abc.ABC):
    __slots__ = ()


class InputScalerBase(ScalerBase, abc.ABC):
    __slots__ = ()
