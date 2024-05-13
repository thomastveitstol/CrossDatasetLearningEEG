import numpy
import torch

from cdl_eeg.data.scalers.scaler_base import TargetScalerBase


# ---------------
# Classes
# ---------------
class ZNormalisation(TargetScalerBase):
    """
    Class for z-normalising
    """

    __slots__ = "_mean", "_std"

    def __init__(self, mean=None, std=None):
        """
        Initialise

        Parameters
        ----------
        mean : float, optional
            Mean. If not provided, it needs to be fit prior to using the transform method
        std : float, optional
            Standard deviation. If not provided, it needs to be fit prior to using the transform method
        """
        self._mean = mean
        self._std = std

    def fit(self, data):
        """
        Method for fitting the parameters of the scaler

        Parameters
        ----------
        data : dict[str, numpy.ndarray]

        Returns
        -------
        None

        Examples
        --------
        >>> my_data = {"d1": numpy.expand_dims(numpy.array([61, 43, 9, 32]), axis=-1),
        ...            "d2": numpy.expand_dims(numpy.array([8, 3, 65, 2, 5, 6]), axis=-1),
        ...            "d3": numpy.expand_dims(numpy.array([7, 2]), axis=-1)}
        >>> my_scaler = ZNormalisation()
        >>> my_scaler.fit(my_data)
        >>> my_scaler.mean, my_scaler.std
        (array([20.25]), array([22.67570286]))
        """
        # Concatenate to a data matrix
        data_matrix = numpy.concatenate(list(data.values()), axis=0)

        # Update parameters
        self._mean = numpy.mean(data_matrix, axis=0)
        self._std = numpy.std(data_matrix, axis=0)

    def transform(self, data):
        """
        Transformation method

        Parameters
        ----------
        data : dict[str, numpy.ndarray]

        Returns
        -------
        dict[str, numpy.ndarray]
            Z-normalised data

        Examples
        --------
        >>> my_fit_data = {"d1": numpy.expand_dims(numpy.array([61, 43, 9, 32]), axis=-1),
        ...                "d2": numpy.expand_dims(numpy.array([8, 3, 65, 2, 5, 6]), axis=-1),
        ...                "d3": numpy.expand_dims(numpy.array([7, 2]), axis=-1)}
        >>> my_scaler = ZNormalisation()
        >>> my_scaler.fit(my_fit_data)
        >>> my_test_data = {"d4": numpy.expand_dims(numpy.array([20.25, 34, 3]), axis=-1),
        ...                 "d5": numpy.expand_dims(numpy.array([54, 4, 22, 7, 103]), axis=-1)}
        >>> my_transformed_data = my_scaler.transform(my_test_data)
        >>> {my_n: numpy.round(my_y, 2) for my_n, my_y in my_transformed_data.items()}  # type: ignore[attr-defined]
        ... # doctest: +NORMALIZE_WHITESPACE
        {'d4': array([[ 0.  ], [ 0.61], [-0.76]]),
         'd5': array([[ 1.49], [-0.72], [ 0.08], [-0.58], [ 3.65]])}
        """
        return {dataset_name: (x - self._mean) / self._std for dataset_name, x in data.items()}

    def inv_transform(self, scaled_data):
        """
        Examples
        --------
        >>> my_fit_data = {"d1": numpy.expand_dims(numpy.array([61, 43, 9, 32]), axis=-1),
        ...                 "d2": numpy.expand_dims(numpy.array([8, 3, 65, 2, 5, 6]), axis=-1),
        ...                 "d3": numpy.expand_dims(numpy.array([7, 2]), axis=-1)}
        >>> my_scaler = ZNormalisation()
        >>> my_scaler.fit(my_fit_data)
        >>> my_test_data = {"d4": numpy.expand_dims(numpy.array([20.25, 34, 3]), axis=-1),
        ...                 "d5": numpy.expand_dims(numpy.array([54, 4, 22, 7, 103]), axis=-1)}
        >>> my_outputs = my_scaler.inv_transform(my_scaler.transform(my_test_data))
        >>> my_outputs  # doctest: +NORMALIZE_WHITESPACE
        {'d4': array([[20.25], [34.  ], [ 3.  ]]),
         'd5': array([[ 54.], [  4.], [ 22.], [  7.], [103.]])}
        """
        if isinstance(scaled_data, dict):
            return {dataset_name: z * self._std + self._mean for dataset_name, z in scaled_data.items()}
        elif isinstance(scaled_data, (numpy.ndarray, torch.Tensor)):
            return scaled_data * self._std + self._mean
        else:
            raise TypeError(f"Unrecognised data type: {type(scaled_data)}")

    # ----------------
    # Properties
    # ----------------
    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


class NoScaler(TargetScalerBase):
    """
    Class for not scaling anything
    """

    __slots__ = ()

    def fit(self, *args, **kwargs):
        ...

    def transform(self, data):
        return data

    def inv_transform(self, scaled_data):
        return scaled_data


# ---------------
# Functions
# ---------------
def get_target_scaler(scaler, **kwargs):
    """
    Function for getting the specified target scaler

    Parameters
    ----------
    scaler : str
    kwargs

    Returns
    -------
    TargetScalerBase
    """
    # All available scalers must be included here
    available_scalers = (ZNormalisation, NoScaler)

    # Loop through and select the correct one
    for target_scaler in available_scalers:
        if scaler == target_scaler.__name__:
            return target_scaler(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The target scaler '{scaler}' was not recognised. Please select among the following: "
                     f"{tuple(target_scaler.__name__ for target_scaler in available_scalers)}")
