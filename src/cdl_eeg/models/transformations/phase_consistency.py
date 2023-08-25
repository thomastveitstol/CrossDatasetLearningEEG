"""
Transformations designed for pretext tasks targeting phase consistency
"""
import random

import numpy
from scipy.signal import hilbert

from cdl_eeg.models.transformations.base import TransformationBase, transformation_method
from cdl_eeg.models.transformations.utils import UnivariateUniform, UnivariateNormal, chunk_eeg, RandomBase

# todo: add EEGChunk as type


class BivariateTimeShift(TransformationBase):
    """
    Transformation class for Phase shifting on chunked EEG, when there are only two channels, by simply shifting the
    time parameter

    Examples:
    ----------
    >>> _ = BivariateTimeShift(2, "normal", num_chunks=3, chunk_duration=300, chunk_time_delay=200)  # Normal
    ... # distribution with std=2 and mean=0
    >>> _ = BivariateTimeShift(6, "uniform", num_chunks=5, chunk_duration=300, chunk_time_delay=200)  # Uniform
    ... # distribution with low=-3 and high=3
    """

    __slots__ = "_distribution", "_num_chunks", "_chunk_duration", "_chunk_time_delay"

    def __init__(self, dist_param, distribution, *, num_chunks, chunk_duration, chunk_time_delay):
        """
        Initialisation method

        Parameters
        ----------
        dist_param : float
            The parameter of the distribution. If distribution is normal, this parameter is standard deviation. If
            uniform distribution, it is the difference between upper and lower bound (with zero mean)
        distribution : str
            Distribution of which to sample the phase shift from. Either 'uniform' or 'normal'
        num_chunks : int
            Number of EEG chunks (see chunk_eeg)
        chunk_duration : int
            Duration of each EEG chunk (see chunk_eeg)
        chunk_time_delay : int
            Duration between each EEG chunk (see chunk_eeg)

        """
        # ----------------
        # Input checks
        # ----------------
        self._check_input(dist_param=dist_param, distribution=distribution)

        # ----------------
        # Set attributes
        # ----------------
        # For the permutation
        sample_distribution: RandomBase
        if distribution == "uniform":
            high = dist_param / 2
            sample_distribution = UnivariateUniform(lower=-high, upper=high)  # Using mean = 0
        elif distribution == "normal":
            sample_distribution = UnivariateNormal(mean=0, std=dist_param)
        else:
            raise AssertionError("This should never happen, please contact the developers")

        self._distribution = sample_distribution

        # Hyperparameters of the chunking of EEG
        self._num_chunks = num_chunks
        self._chunk_duration = chunk_duration
        self._chunk_time_delay = chunk_time_delay

    @staticmethod
    def _check_input(dist_param, distribution):
        # Check that the distribution is recognised
        legal_distributions = ("uniform", "normal")
        if distribution not in legal_distributions:
            raise ValueError(f"Invalid distribution input value. Expected one of the following: {distribution}, but "
                             f"received '{legal_distributions}'")

        # Check that the parameter is greater than zero
        if dist_param <= 0:
            param_name = "standard deviation" if distribution == "normal" else "lower/upper bound width"
            raise ValueError(f"Expected {param_name} to be greater than zero, but found {dist_param}")

    # ----------------------
    # Transformation methods
    # ----------------------
    @transformation_method
    def phase_shift(self, x0, x1, permute_first_channel):
        """
        Transformation by shifting (permuting the time parameter of) a single channel in a single EEG chunk

        Parameters
        ----------
        x0 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps) or shape=(batch, 1, time_steps)
        x1 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps) or shape=(batch, 1, time_steps)
        permute_first_channel : bool

        Returns
        -------
        tuple[tuple[numpy.ndarray, ...], int, float]
            A tuple, containing the EEG chunks (one is permuted by phase shift in one of the channels), the index
            indicating which chunk was permuted, and the phase shift value
        """
        # ----------------
        # Input checks
        # ----------------
        # Dimensions of the EEG channels
        if x0.ndim != 2 or x1.ndim != 2:
            raise ValueError(f"Expected the EEG channels to have two dimensions, but received {x0.ndim} and {x1.ndim}")

        # Shape of the EEG channels
        if x0.shape != x1.shape:
            raise ValueError(f"Expected the EEG channels to have the same shape, but received {x0.shape} and "
                             f"{x1.shape}")

        # Type check
        if not isinstance(permute_first_channel, bool):
            raise TypeError(f"Expected 'permute_first_channel' to be boolean, but received "
                            f"{type(permute_first_channel)}")

        # ----------------
        # Chunk the EEG
        # ----------------
        # Concatenate the EEG signals. The resulting bivariate EEG will have shape=(batch, 2, time_steps)
        data = numpy.concatenate((numpy.expand_dims(x0, axis=1), numpy.expand_dims(x1, axis=1)), axis=1)

        # Create non-permuted EEG chunks
        eeg_chunks = chunk_eeg(data=data, k=self._num_chunks, chunk_duration=self._chunk_duration,
                               delta_t=self._chunk_time_delay, chunk_start_shift=0)

        # Create permuted EEG chunks  todo: this is not the most memory-efficient way...
        phase_shift = int(self._distribution.draw())  # todo: needs to be an integer, consider mapping to seconds
        phase_shifted_chunks = chunk_eeg(data=data, k=self._num_chunks, chunk_duration=self._chunk_duration,
                                         delta_t=self._chunk_time_delay, chunk_start_shift=phase_shift)

        # ----------------
        # Replace the specified channel in a randomly
        # selected chunk by its permuted version
        # ----------------
        # Generate random chunk
        permuted_chunk = random.randint(0, self._num_chunks-1)

        # Perform permutation
        permuted_channel = 0 if permute_first_channel else 1
        eeg_chunks[permuted_chunk][:, permuted_channel] = phase_shifted_chunks[permuted_chunk][:, permuted_channel]

        # ----------------
        # Return permuted chunks, index of permuted chunk,
        # and phase shift
        # ----------------
        return eeg_chunks, permuted_chunk, phase_shift


class BivariatePhaseShift(TransformationBase):
    """
    Transformation class for Phase shifting on chunked EEG, when there are only two channels, by shifting the phase

    Examples:
    ----------
    >>> _ = BivariatePhaseShift(UnivariateUniform(0, 2*numpy.pi), num_chunks=5, chunk_duration=2000,
    ...                         chunk_time_delay=1000)
    """

    __slots__ = "_phase_shift_distribution", "_num_chunks", "_chunk_duration", "_chunk_time_delay"

    def __init__(self, phase_shift_distribution, *, num_chunks, chunk_duration, chunk_time_delay):
        """
        Initialisation method

        Parameters
        ----------
        phase_shift_distribution : RandomBase
            Distribution of which to sample the phase shift from when performing the permutation
        num_chunks : int
            Number of EEG chunks (see chunk_eeg)
        chunk_duration : int
            Duration of each EEG chunk (see chunk_eeg)
        chunk_time_delay : int
            Duration between each EEG chunk (see chunk_eeg)
        """
        # ----------------
        # Input checks
        # ----------------
        self._check_phase_shift_distribution(phase_shift_distribution)

        # ----------------
        # Set attributes
        # ----------------
        # For the permutation
        self._phase_shift_distribution = phase_shift_distribution

        # For chunking of EEG
        self._num_chunks = num_chunks
        self._chunk_duration = chunk_duration
        self._chunk_time_delay = chunk_time_delay

    @staticmethod
    def _check_phase_shift_distribution(phase_shift_distribution):
        # Type check
        if not isinstance(phase_shift_distribution, RandomBase):
            raise TypeError(f"Expected phase shift distribution to be an instance of 'RandomBase', but found "
                            f"{type(phase_shift_distribution)}")

    # ----------------------
    # Transformation methods
    # ----------------------
    @transformation_method
    def phase_shift(self, x0, x1, permute_first_channel):
        """
        Transformation by shifting the phase of a single channel in a single EEG chunk

        Parameters
        ----------
        x0 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps)
        x1 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps)
        permute_first_channel : bool

        Returns
        -------
        tuple[tuple[numpy.ndarray, ...], int, float]
            A tuple, containing the EEG chunks (one is permuted by phase shift in one of the channels), the index
            indicating which chunk was permuted, and the phase shift value
        """
        # ----------------
        # Input checks  todo: too similar to the other class
        # ----------------
        # Dimensions of the EEG channels
        if x0.ndim != 2 or x1.ndim != 2:
            raise ValueError(f"Expected the EEG channels to have two dimensions, but received {x0.ndim} and {x1.ndim}")

        # Shape of the EEG channels
        if x0.shape != x1.shape:
            raise ValueError(f"Expected the EEG channels to have the same shape, but received {x0.shape} and "
                             f"{x1.shape}")

        # Type check
        if not isinstance(permute_first_channel, bool):
            raise TypeError(f"Expected 'permute_first_channel' to be boolean, but received "
                            f"{type(permute_first_channel)}")

        # ----------------
        # Chunk the EEG
        # ----------------
        # Concatenate the EEG signals. The resulting bivariate EEG will have shape=(batch, 2, time_steps)
        data = numpy.concatenate((numpy.expand_dims(x0, axis=1), numpy.expand_dims(x1, axis=1)), axis=1)

        # Create non-permuted EEG chunks  todo: to copy or not to copy the numpy data
        eeg_chunks = chunk_eeg(data=data, k=self._num_chunks, chunk_duration=self._chunk_duration,
                               delta_t=self._chunk_time_delay, chunk_start_shift=0)

        # ----------------
        # Create permutation
        # ----------------
        # Apply Hilbert transformation to the selected channel and chunk to obtain the analytic signal and phase
        permuted_chunk = random.randint(0, self._num_chunks - 1)  # Selected chunk
        channel = 0 if permute_first_channel else 1
        analytic_signal = hilbert(eeg_chunks[permuted_chunk][:, channel])
        phase_data = numpy.angle(analytic_signal)

        # Alter the phase
        phase_shift = self._phase_shift_distribution.draw()
        phase_data += phase_shift

        # Compute the modified data and insert it
        modified_eeg_data = numpy.real(numpy.abs(analytic_signal) * numpy.exp(1j * phase_data))
        eeg_chunks[permuted_chunk][:, channel] = modified_eeg_data

        # ----------------
        # Return permuted chunks, index of permuted chunk,
        # and phase shift
        # ----------------
        return eeg_chunks, permuted_chunk, phase_shift
