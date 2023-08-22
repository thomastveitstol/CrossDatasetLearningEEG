import abc

import numpy.random


def chunk_eeg(data, *, k, chunk_duration, delta_t):
    """Function for splitting EEG data into chunks

    (unittest in test folder)
    Parameters
    ----------
    data : numpy.ndarray
        EEG data with shape=(num_subjects, num_channels, num_timesteps)
    k : int
        Number of EEG chunks
    chunk_duration : int
        The duration of the EEG chunks (in number of timesteps)
    delta_t : int
        time duration between the chunks, in number of timesteps

    Returns
    -------
    tuple[numpy.ndarray, ...]
        The EEG data split into chunks

    """
    # Extracting EEG chunks 'from the centre' is achieved by starting at residue / 2, where residue is the difference
    # between total number of time steps of the EEG data and the amount of EEG data used in the chunks
    expected_used_eeg = chunk_duration * k + delta_t * (k - 1)
    total_num_timesteps = data.shape[-1]

    assert total_num_timesteps >= expected_used_eeg, (f"The specified hyperparameters require EEG data with "
                                                      f"{expected_used_eeg} number of time steps, but only "
                                                      f"{total_num_timesteps} are available.")
    chunk_start = (total_num_timesteps - expected_used_eeg) // 2

    # Add the first chunk
    chunk = [data[..., chunk_start:(chunk_start + chunk_duration)]]

    # Append all the other chunks
    for i in range(1, k):
        i0 = i * (delta_t + chunk_duration) + chunk_start
        chunk.append(data[..., i0:(i0+chunk_duration)])

    return tuple(chunk)


# -------------------------
# Random number classes
# -------------------------
class RandomBase(abc.ABC):

    @abc.abstractmethod
    def draw(self, seed=None):
        """
        Draw a sample from the distribution

        Parameters
        ----------
        seed : int
            A seed can be passed for reproducibility

        Returns
        -------
        A sample from the distribution
        """


class UnivariateNormal(RandomBase):
    """
    Class for drawing samples from a univariate normal distribution. To reproduce, numpy.random.seed must be called
    """

    __slots__ = "_mean", "_std"

    def __init__(self, mean=0, std=1):
        """
        Initialisation method

        Parameters
        ----------
        std : float
            Standard deviation of the normal distribution
        mean : float
            Mean of the normal distribution
        """
        # ------------------
        # Set attributes
        # ------------------
        self._mean = mean
        self._std = std

    def draw(self, seed=None):
        """
        Examples
        -------
        >>> UnivariateNormal(3, 0.5).draw(seed=1)  # doctest: +ELLIPSIS
        3.812...

        It is the same to set seed outside, as passing the seed to the method

        >>> numpy.random.seed(1)
        >>> UnivariateNormal(3, 0.5).draw()  # doctest: +ELLIPSIS
        3.812...
        """
        # Maybe set seed
        if seed is not None:
            numpy.random.seed(seed)

        # Draw a sample from the distribution and return
        return numpy.random.normal(loc=self._mean, scale=self._std)


class UnivariateUniform(RandomBase):

    __slots__ = "_lower", "_upper"

    def __init__(self, lower, upper):
        """
        Initialisation

        Parameters
        ----------
        lower : float
            Lower bound for the uniform distribution
        upper : float
            Upper bound for the uniform distribution
        """
        # ------------------
        # Set attributes
        # ------------------
        self._lower = lower
        self._upper = upper

    def draw(self, seed=None):
        """
        Examples:
        >>> UnivariateUniform(-1, 1).draw(seed=1)  # doctest: +ELLIPSIS
        -0.165...

        It is the same to set seed outside, as passing the seed to the method

        >>> numpy.random.seed(1)
        >>> UnivariateUniform(-1, 1).draw()  # doctest: +ELLIPSIS
        -0.165...
        """
        # Maybe set seed
        if seed is not None:
            numpy.random.seed(seed)

        # Draw a sample from the distribution and return
        return numpy.random.uniform(low=self._lower, high=self._upper)
