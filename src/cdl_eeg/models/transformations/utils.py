import abc

import mne
import numpy.random


def chunk_eeg(data, *, k, chunk_duration, delta_t, chunk_start_shift=0):
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
    chunk_start_shift : int
        The first chunk is started such that the chunks are centred. However, this may be changed på this parameter
        todo: make/improve input checks

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
    chunk_start = (total_num_timesteps - expected_used_eeg) // 2 + chunk_start_shift

    # Add the first chunk
    chunk = [data[..., chunk_start:(chunk_start + chunk_duration)]]

    # Append all the other chunks
    for i in range(1, k):
        i0 = i * (delta_t + chunk_duration) + chunk_start
        chunk.append(data[..., i0:(i0+chunk_duration)])

    return tuple(chunk)


def eeg_chunks_to_mne_epochs(chunks, info=None, *, sampling_freq=None, ch_names=None, verbose=None):
    """
    Convert from EEG numpy chunks to MNE epochs object.
    Parameters
    ----------
    chunks : tuple[numpy.ndarray, ...]
        Chunks of EEG, as returned by chunk_eeg function
    info : mne.Info, optional
    sampling_freq : float, optional
        Sampling frequency. Required if 'info' argument is not passed, ignored if 'info' is passed
    ch_names : tuple[str, ...], optional
        Channel names. Ignored if 'info' is specified. If None is passed, the channel names will be ('Ch1', 'Ch2', ...)
    verbose : bool, optional
        To print from the MNE operations (True) or not (False)

    Returns
    -------
    mne.EpochsArray
        mne.EpochsArray object with the chunked EEG
    """
    # (Maybe) create info object
    if info is None:
        # Maybe set default channel names
        ch_names = chunks[0].shape[0] if ch_names is None else list(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_freq, verbose=verbose)

    # Create EpochsArray object and return
    return mne.EpochsArray(data=numpy.array(chunks), info=info, verbose=verbose)


# -------------------------
# Random number classes
# -------------------------
class RandomBase(abc.ABC):

    __slots__ = ()

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
        float
            A sample from the distribution
        """

    @abc.abstractmethod
    def scale(self, draw):
        """
        Method for scaling a random  draw

        Parameters
        ----------
        draw : float

        Returns
        -------
        float
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

    def scale(self, draw):
        """
        Scaling method which subtracts the mean and divide by standard deviation

        Parameters
        ----------
        draw : float

        Returns
        -------
        float

        Examples
        --------
        >>> UnivariateNormal(3, 0.5).scale(3.5)
        1.0
        >>> round(UnivariateNormal(3, 0.5).scale(2.7), 5)
        -0.6
        """
        return (draw - self._mean) / self._std


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

    def scale(self, draw):
        """
        Scales such that the distribution is U[-1, 1]

        Parameters
        ----------
        draw : float

        Returns
        -------
        float

        Examples
        --------
        >>> UnivariateUniform(9, 10).scale(9)
        -1.0
        >>> UnivariateUniform(9, 10).scale(10)
        1.0
        >>> UnivariateUniform(9, 10).scale(9.5)
        0.0
        >>> UnivariateUniform(9, 10).scale(9.75)
        0.5
        """
        # Subtract mean
        draw -= (self._upper + self._lower) / 2

        # Scale interval
        draw /= (self._upper - self._lower) / 2

        return draw

    # --------------
    # Properties
    # --------------
    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper


# -------------------------
# Functions
# -------------------------
def get_random_distribution(distribution, **kwargs):
    """
    Function for getting the specified random distribution

    Parameters
    ----------
    distribution : str
    kwargs

    Returns
    -------
    RandomBase
    """
    # All available distributions must be included here
    available_distributions = (UnivariateUniform, UnivariateNormal)

    # Loop through and select the correct one
    for dist in available_distributions:
        if distribution == dist.__name__:
            return dist(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The random distribution '{distribution}' was not recognised. Please select among the following: "
                     f"{tuple(dist.__name__ for dist in available_distributions)}")
