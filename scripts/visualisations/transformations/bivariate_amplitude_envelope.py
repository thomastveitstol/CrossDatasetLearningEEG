"""
Script for plotting the results of the Bivariate Amplitude Envelope transform

todo: too long and similar to the original visualisation. Start making the Dataset classes
"""
import contextlib
import io
import os

import mne
import numpy
from matplotlib import pyplot

from cdl_eeg.data.paths import get_raw_data_storage_path
from cdl_eeg.models.transformations.amplitude_envelope import BivariateAmplitudeEnvelopePermutation
from cdl_eeg.models.transformations.utils import chunk_eeg, eeg_chunks_to_mne_epochs, UnivariateNormal, \
    UnivariateUniform


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Chunking
    num_chunks = 10
    chunk_duration = 2000
    chunk_time_delay = 1000
    ch_0, ch_1 = 0, 7

    # Permutation
    envelope_std = UnivariateNormal(250, 50)
    envelope_amp = UnivariateUniform(0.3, 0.7)

    # Data
    subject = 24
    derivatives = True  # boolean, indicating cleaned/not cleaned data

    # Preprocessing
    l_freq = 1
    h_freq = 45

    verbose = False

    # -----------------
    # Load data
    # -----------------
    # Make path
    dataset_path = os.path.join(get_raw_data_storage_path(), "miltiadous")
    dataset_path = os.path.join(dataset_path, "derivatives") if derivatives else dataset_path

    subject_id = tuple(dataset for dataset in os.listdir(dataset_path) if dataset[:4] == "sub-")[subject]
    path = os.path.join(dataset_path, subject_id, "eeg", f"{subject_id}_task-eyesclosed_eeg.set")

    # Make MNE raw object
    eeg: mne.io.eeglab.eeglab.RawEEGLAB = mne.io.read_raw_eeglab(input_fname=path, preload=True, verbose=verbose)

    # -----------------
    # Some additional preprocessing steps
    # -----------------
    # Filtering
    eeg.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # Re-referencing
    eeg.set_eeg_reference(ref_channels="average", verbose=verbose)

    # -----------------
    # Split into chunks
    # -----------------
    # Extract two channels as numpy array (with batch dimension)
    data = numpy.expand_dims(eeg.get_data()[[ch_0, ch_1]], axis=0)
    data = (data - numpy.mean(data, axis=-1, keepdims=True)) / (numpy.std(data, axis=-1, keepdims=True) + 1e-8)

    # Chunk, both original and permuted
    original_chunks = chunk_eeg(data, k=num_chunks, chunk_duration=chunk_duration, delta_t=chunk_time_delay)

    transformation = BivariateAmplitudeEnvelopePermutation(envelope_std=envelope_std, envelope_amp=envelope_amp,
                                                           num_chunks=num_chunks, chunk_duration=chunk_duration,
                                                           chunk_time_delay=chunk_time_delay)
    permuted_chunks, idx, t0, sigma, amp = transformation.gaussian_envelope_permutation(x0=data[:, 0], x1=data[:, 1],
                                                                                        permute_first_channel=True)

    # -----------------
    # Convert to MNE objects, plot, and print details
    # -----------------
    # Need to remove batch dimensions
    original_chunks = tuple(numpy.squeeze(chunk, axis=0) for chunk in original_chunks)
    permuted_chunks = tuple(numpy.squeeze(chunk, axis=0) for chunk in permuted_chunks)

    # Create MNE objects
    original_epochs = eeg_chunks_to_mne_epochs(original_chunks, sampling_freq=eeg.info["sfreq"], verbose=verbose)
    permuted_epochs = eeg_chunks_to_mne_epochs(permuted_chunks, sampling_freq=eeg.info["sfreq"], verbose=verbose)

    # Print permuted chunk and phase shift
    print("\n----- Permutation details -----")
    print(f"Permuted chunk (index): {idx}")
    print(f"Time t0: {t0 / eeg.info['sfreq']:.2f}s")
    print(f"Sigma: {sigma / eeg.info['sfreq']:.2f}s")
    print(f"Amplitude: {amp:.2f}\n")

    # Plot
    if verbose:
        original_epochs.plot(picks="all", title="Original", events=False, scalings=15)
        permuted_epochs.plot(picks="all", title="Permuted", events=False, scalings=15)

        pyplot.show()
    else:
        # Redirect to an unused StringIO object
        with contextlib.redirect_stdout(io.StringIO()):
            original_epochs.plot(picks="all", title="Original", events=False, scalings=15)
            permuted_epochs.plot(picks="all", title="Permuted", events=False, scalings=15)

            pyplot.show()


if __name__ == "__main__":
    main()
