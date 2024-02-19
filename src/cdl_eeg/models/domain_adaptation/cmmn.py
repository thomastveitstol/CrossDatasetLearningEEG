"""
Implementation of Convolutional Monge Mapping Normalisation.

Original implementation:  todo: check their LICENSE.txt
    https://github.com/PythonOT/convolutional-monge-mapping-normalization/tree/main

Paper:
    T. Gnassounou, R. Flamary, A. Gramfort, Convolutional Monge Mapping Normalization for learning on biosignals,
    Neural Information Processing Systems (NeurIPS), 2023.
"""
from typing import Dict, Optional

import numpy
from scipy import fft
from scipy import signal


class ConvMMN:
    """
    Implementation of CMMN
    """

    __slots__ = "_sampling_freq", "_kernel_size", "_psd_barycenter", "_monge_filter"

    def __init__(self, *, kernel_size, sampling_freq=None):
        """Initialise object"""
        # Store attributes
        self._sampling_freq = sampling_freq
        self._kernel_size = kernel_size

        # Initialise attributes to be fitted
        self._psd_barycenter: Optional[numpy.ndarray] = None
        self._monge_filter: Dict[str, numpy.ndarray] = dict()

    # ---------------
    # Methods for fitting CMMN
    # ---------------
    def fit_barycenter(self, data, *, sampling_freq=None):
        # If a sampling frequency is passed, set it (currently overriding, may want to do a similarity check and raise
        # error)
        self._sampling_freq = self._sampling_freq if sampling_freq is None else sampling_freq
        if self._sampling_freq is None:
            raise ValueError("A sampling frequency must be provided either in the __init__ or fit method, but None was "
                             "found")

        # -------------
        # Compute the PSD barycenter
        # -------------
        # Compute representative PSDs of all datasets
        psds = self._compute_representative_psds(data=data, sampling_freq=self._sampling_freq,
                                                 kernel_size=self._kernel_size)

        # Compute PSD barycenter
        self._psd_barycenter = self._compute_psd_barycenter(psds)

    def fit_monge_filters(self, data):
        """
        Method for fitting the filter used for monge mapping (fitting h_k in the paper). With this implementation, a
        monge filter is fit per dataset. This may change in the future.

        Sampling frequency should be the same as was used during fitting, although no checks are made

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            The data for which monge filters will be fitted. Keys are dataset names, values are EEG numpy arrays with
            shape=(subjects, channels, time_steps)

        Returns
        -------
        None
        """
        # Checks
        if self._psd_barycenter is None:
            raise RuntimeError("The barycenter must be fit before fitting the monge filters")

        # Loop through all provided datasets
        for dataset_name, x in data.items():
            # Compute representative PSD
            dataset_psd = self._compute_representative_psd(x, sampling_freq=self._sampling_freq,
                                                           kernel_size=self._kernel_size)

            # Compute the monge filter as in Eq. 5  todo: verify that the axes is correct
            monge_filter = fft.irfftn(numpy.sqrt(self._psd_barycenter) / numpy.sqrt(dataset_psd), axes=(-1,))

            # Store the monge filter
            self._monge_filter[dataset_name] = monge_filter

    # ---------------
    # Methods for computing PSDs
    # ---------------
    @staticmethod
    def _compute_single_source_psd(x, *, sampling_freq, kernel_size):
        r"""
        Compute the PSDs for a single source domain. If averaged afterward, it will be \hat{\textbf{p}}_k in the
        original paper

        Parameters
        ----------
        x : numpy.ndarray
            The EEG data of a single dataset. Should have shape=(num_subjects, channels, time_steps)
        sampling_freq : float
        kernel_size : int
            This is called 'filter_size' in the original implementation by the authors, and is the value 'F' in the
            original paper (at least, the pre-print which is the only currently available version)

        Returns
        -------
        numpy.ndarray
            The PSDs of all subjects and channels

        Examples
        --------
        >>> my_data = numpy.random.normal(0, 1, size=(10, 32, 2000))
        >>> my_psds = ConvMMN._compute_single_source_psd(my_data, sampling_freq=10, kernel_size=128)
        >>> my_psds.shape
        (10, 32, 65)
        """
        # todo: I don't really understand nperseg
        return signal.welch(x=x, axis=-1, fs=sampling_freq, nperseg=kernel_size)[-1]

    @staticmethod
    def _aggregate_subject_psds(x):
        """
        Aggregate the PSDs computed on a single dataset.

        Parameters
        ----------
        x : numpy.ndarray
            Should have shape=(num_subjects, channels, frequencies)

        Returns
        -------
        numpy.ndarray
            Will have shape=(channels, frequencies)

        Examples
        --------
        >>> my_psds = numpy.random.normal(0, 1, size=(10, 32, 65))
        >>> ConvMMN._aggregate_subject_psds(my_psds).shape
        (32, 65)
        """
        # todo: Would be cool to support different aggregation methods, and in particular to support different weights
        #  for different sub-groups
        return numpy.mean(x, axis=0)

    @classmethod
    def _compute_representative_psd(cls, x, *, sampling_freq, kernel_size):
        # Compute PSDs and aggregate them
        return cls._aggregate_subject_psds(cls._compute_single_source_psd(x=x, sampling_freq=sampling_freq,
                                                                          kernel_size=kernel_size))

    @classmethod
    def _compute_representative_psds(cls, data, *, sampling_freq, kernel_size):
        """
        Method for computing representative PSDs of multiple datasets

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            The EEG data from all datasets. The keys are dataset names
        sampling_freq : float
        kernel_size : int

        Returns
        -------
        dict[str, numpy.ndarray]
            The keys are dataset names, the values are PSDs

        Examples
        --------
        >>> my_data = {"d1": numpy.random.normal(0, 1, size=(10, 32, 2000)),
        ...            "d2": numpy.random.normal(0, 1, size=(8, 64, 1500)),
        ...            "d3": numpy.random.normal(0, 1, size=(17, 19, 3000))}
        >>> my_estimated_psds = ConvMMN._compute_representative_psds(my_data, sampling_freq=100, kernel_size=64)
        >>> {name_: psds_.shape for name_, psds_ in my_estimated_psds.items()}  # type: ignore[attr-defined]
        {'d1': (32, 33), 'd2': (64, 33), 'd3': (19, 33)}
        """
        return {dataset_name: cls._compute_representative_psd(x=x, sampling_freq=sampling_freq, kernel_size=kernel_size)
                for dataset_name, x in data.items()}

    @staticmethod
    def _compute_psd_barycenter(psds):
        """
        Compute the barycenter (eq. 6 in the paper), given the PSDs from the different datasets

        Parameters
        ----------
        psds: dict[str, numpy.ndarray]
            The PSDs. The numpy arrays should have shape=(channels, frequencies). The channel dimension must be the same
            for all datasets

        Returns
        -------
        numpy.ndarray
        """
        # todo: in the future, other aggregation methods and weighting may be implemented
        return numpy.mean(numpy.concatenate([numpy.expand_dims(psd, axis=0) for psd in psds.values()], axis=0), axis=0)
