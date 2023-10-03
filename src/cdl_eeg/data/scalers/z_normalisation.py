import numpy


class ZNormalisation:
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
        std : float, optional
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
        >>> my_data = {"d1": numpy.array([61, 43, 9, 32]), "d2": numpy.array([8, 3, 65, 2, 5, 6]),
        ...            "d3": numpy.array([7, 2])}
        >>> my_scaler = ZNormalisation()
        >>> my_scaler.fit(my_data)
        >>> my_scaler.mean, round(my_scaler.std, 2)
        (20.25, 22.68)
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
        >>> my_fit_data = {"d1": numpy.array([61, 43, 9, 32]), "d2": numpy.array([8, 3, 65, 2, 5, 6]),
        ...                "d3": numpy.array([7, 2])}
        >>> my_scaler = ZNormalisation()
        >>> my_scaler.fit(my_fit_data)
        >>> my_test_data = {"d4": numpy.array([20.25, 34, 3]), "d5": numpy.array([54, 4, 22, 7, 103])}
        >>> my_transformed_data = my_scaler.transform(my_test_data)
        >>> {my_n: numpy.round(my_y, 2) for my_n, my_y in my_transformed_data.items()}  # type: ignore[attr-defined]
        {'d4': array([ 0.  ,  0.61, -0.76]), 'd5': array([ 1.49, -0.72,  0.08, -0.58,  3.65])}
        """
        return {dataset_name: (x - self._mean) / self._std for dataset_name, x in data.items()}

    # ----------------
    # Properties
    # ----------------
    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std
