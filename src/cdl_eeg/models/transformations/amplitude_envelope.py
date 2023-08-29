"""
Transformations designed for pretext tasks targeting amplitude envelope characteristics. See e.g. Amplitude Envelope
Correlation:

Bruns, Andreas; Eckhorn, Reinhard; Jokeit, Hennric; Ebner, Alois. Amplitude envelope correlation detects coupling among
incoherent brain signals. NeuroReport 11(7):p 1509-1514, May 15, 2000.
"""
import random

import numpy
from scipy.signal import hilbert

from cdl_eeg.models.transformations.base import TransformationBase, transformation_method
from cdl_eeg.models.transformations.utils import RandomBase, chunk_eeg


class BivariateAmplitudeEnvelopePermutation(TransformationBase):
    """
    Transformation class for permuting a channel in a single chunk by multiplying it with
    g(t) = 1 - A*exp(-0.5(t-t0)^2 / sigma^(-2))

    Examples:
    ----------
    >>> from cdl_eeg.models.transformations.utils import UnivariateNormal, UnivariateUniform
    >>> _ = BivariateAmplitudeEnvelopePermutation(UnivariateNormal(250, 50), UnivariateUniform(0.3, 0.7), num_chunks=5,
    ...                                           chunk_duration=2000, chunk_time_delay=1000)
    >>> _ = BivariateAmplitudeEnvelopePermutation(UnivariateUniform(170, 330), UnivariateNormal(0.5, 0.1), num_chunks=6,
    ...                                           chunk_duration=21800, chunk_time_delay=500)
    """

    __slots__ = "_envelope_std", "_envelope_amp", "_num_chunks", "_chunk_duration", "_chunk_time_delay"

    def __init__(self, envelope_std, envelope_amp, *, num_chunks, chunk_duration, chunk_time_delay):
        """
        Initialisation method

        Parameters
        ----------
        envelope_std : RandomBase
            Distribution of which to sample the standard deviation sigma from when performing the permutation
        envelope_amp : RandomBase
            Distribution of which to sample the amplitude A from when performing the permutation
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
        self._check_input_distributions(envelope_std, envelope_amp)

        # ----------------
        # Set attributes
        # ----------------
        # For the permutation
        self._envelope_std = envelope_std
        self._envelope_amp = envelope_amp

        # For chunking of EEG
        self._num_chunks = num_chunks
        self._chunk_duration = chunk_duration
        self._chunk_time_delay = chunk_time_delay

    @staticmethod
    def _check_input_distributions(envelope_std, envelope_amp):
        # Type check
        if not isinstance(envelope_std, RandomBase):
            raise TypeError(f"Expected the input standard deviation distribution to be an instance of 'RandomBase', "
                            f"but found {type(envelope_std)}")
        if not isinstance(envelope_amp, RandomBase):
            raise TypeError(f"Expected the input amplitude distribution to be an instance of 'RandomBase', but found "
                            f"{type(envelope_amp)}")

    # ----------------------
    # Transformation methods
    # ----------------------
    @transformation_method
    def gaussian_envelope_permutation(self, x0, x1, permute_first_channel):
        """
        Transformation by multiplying a single chunk and channel by g(t) = 1 - A*exp(-0.5(t-t0)^2 / sigma^(-2))

        Parameters
        ----------
        x0 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps) or shape=(batch, 1, time_steps)
        x1 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps) or shape=(batch, 1, time_steps)
        permute_first_channel : bool
            To permute the first channel (True) or the second one (False)

        Returns
        -------
        tuple[tuple[numpy.ndarray, ...], int, float, float, float]
        """
        # ----------------
        # Input checks  todo: copied from BivariatePhaseShift
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
        # Chunk the EEG and make a permutation
        # ----------------
        # Concatenate the EEG signals. The resulting bivariate EEG will have shape=(batch, 2, time_steps)
        data = numpy.concatenate((numpy.expand_dims(x0, axis=1), numpy.expand_dims(x1, axis=1)), axis=1)

        # Create EEG chunks
        eeg_chunks = chunk_eeg(data=data, k=self._num_chunks, chunk_duration=self._chunk_duration,
                               delta_t=self._chunk_time_delay, chunk_start_shift=0)

        # Select random chunk
        permuted_chunk = random.randint(0, self._num_chunks - 1)

        # Generate envelope permutation
        chunk_length = eeg_chunks[permuted_chunk].shape[-1]  # Number of time steps in the selected chunk
        t = numpy.arange(chunk_length)
        t0 = numpy.random.uniform(low=0, high=chunk_length)  # todo: consider changing this
        sigma = self._envelope_std.draw()
        amplitude = self._envelope_amp.draw()

        envelope = 1 - amplitude * numpy.exp(-0.5 * (t - t0) ** 2 / (sigma ** 2))

        # Perform permutation
        channel = 0 if permute_first_channel else 1
        eeg_chunks[permuted_chunk][:, channel] *= envelope

        # ----------------
        # Return chunks, index of permuted chunk, t0, sigma, and amplitude
        # ----------------
        return eeg_chunks, permuted_chunk, t0, sigma, amplitude


class BivariateAmplitudeEnvelopeTimeShift(TransformationBase):
    """
    Transformation where the amplitude envelope is shifted in time
    """

    __slots__ = "_time_shift", "_num_chunks", "_chunk_duration", "_chunk_time_delay"

    def __init__(self, time_shift, *, num_chunks, chunk_duration, chunk_time_delay):
        """
        Initialisation method

        Parameters
        ----------
        time_shift : RandomBase
            Distribution of which to sample the time shift from
        num_chunks : int
            Number of EEG chunks (see chunk_eeg)
        chunk_duration : int
            Duration of each EEG chunk (see chunk_eeg)
        chunk_time_delay : int
            Duration between each EEG chunk (see chunk_eeg)
        """
        # ----------------
        # Set attributes
        # ----------------
        # For the permutation
        self._time_shift = time_shift

        # For chunking of EEG
        self._num_chunks = num_chunks
        self._chunk_duration = chunk_duration
        self._chunk_time_delay = chunk_time_delay

    # ----------------------
    # Transformation methods
    # ----------------------
    @transformation_method
    def time_shift(self, x0, x1, permute_first_channel):
        """
        Transformation by time shifting of the amplitude envelope

        todo: unittest
        Parameters
        ----------
        x0 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps) or shape=(batch, 1, time_steps)
        x1 : numpy.ndarray
            Single channel EEG, with shape=(batch, time_steps) or shape=(batch, 1, time_steps)
        permute_first_channel : bool
            To permute the first channel (True) or the second one (False)

        Returns
        -------
        tuple[tuple[numpy.ndarray, ...], int, int]
            Permuted chunks, index of permuted chunk, and time shift
        """
        # ----------------
        # Input checks  todo: copied from BivariateAmplitudeEnvelopePermutation
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
        time_shift = int(self._time_shift.draw())
        time_shifted_chunks = chunk_eeg(data=data, k=self._num_chunks, chunk_duration=self._chunk_duration,
                                        delta_t=self._chunk_time_delay, chunk_start_shift=time_shift)

        # ----------------
        # Replace the amplitude of the specified channel in a randomly
        # selected chunk by its permuted version
        # ----------------
        # Apply Hilbert transformation to the selected channel and chunk to obtain the analytic signal and amplitude
        permuted_chunk = random.randint(0, self._num_chunks - 1)  # Selected chunk
        channel = 0 if permute_first_channel else 1

        original_analytic_signal = hilbert(eeg_chunks[permuted_chunk][:, channel])
        original_phase = numpy.angle(original_analytic_signal)

        shifted_analytic_signal = hilbert(time_shifted_chunks[permuted_chunk][:, channel])
        shifted_amplitude_envelope = numpy.abs(shifted_analytic_signal)

        # Insert a new signal where the phase is maintained, but the amplitude envelope is time shifted
        modified_signal = numpy.real(numpy.abs(shifted_amplitude_envelope) * numpy.exp(1j * original_phase))
        eeg_chunks[permuted_chunk][:, channel] = modified_signal

        # ----------------
        # Return permuted chunks, index of permuted chunk,
        # and time shift
        # ----------------
        return eeg_chunks, permuted_chunk, time_shift
