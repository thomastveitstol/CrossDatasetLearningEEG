import numpy
from scipy.signal import hilbert

from cdl_eeg.models.transformations.base import TransformationBase, transformation_method
from cdl_eeg.models.transformations.utils import RandomBase


class FrequencySlowing(TransformationBase):
    """
    Transformation where the frequency/phase is slowed down (or sped up)

    Examples
    --------
    >>> from cdl_eeg.models.transformations.utils import UnivariateNormal
    >>> _ = FrequencySlowing(UnivariateNormal(1, 0.05))
    """

    __slots__ = "_slowing_distribution",

    def __init__(self, slowing_distribution):
        """
        Initialise

        Parameters
        ----------
        slowing_distribution : RandomBase
            Distribution of which to sample the slowing of frequency from
        """
        # Set attribute
        if not isinstance(slowing_distribution, RandomBase):
            raise TypeError(f"Expected the input distribution to be an instance of 'RandomBase', but found "
                            f"{type(slowing_distribution)}")
        self._slowing_distribution = slowing_distribution

    # ----------------------
    # Transformation methods
    # ----------------------
    @transformation_method
    def phase_slowing(self, x):
        """
        Slows down the phase of the signal, but keeps the amplitude envelope

        Parameters
        ----------
        x : numpy.ndarray
            EEG data with shape=(batch, channels, time_steps)

        Returns
        -------
        tuple[numpy.ndarray, float]
            Both the slowed down EEG data and how much it was slowed down
        """
        # Apply Hilbert transformation
        analytical_signal = hilbert(x)

        # Keep the amplitude envelope
        amplitude_envelope = numpy.abs(analytical_signal)

        # Slow down the phase by multiplication
        phase_modulation = self._slowing_distribution.draw()
        modified_phase = numpy.angle(analytical_signal ** phase_modulation)  # todo: ask Mia/Ricardo if this is correct

        # (Create and) return the new signal and phase modulation
        return numpy.real(amplitude_envelope * numpy.exp(1j * modified_phase)), phase_modulation
