"""
Script for plotting the results of the Bivariate Amplitude Envelope Time Shift transform

todo: too long and similar to the original visualisation. Start making the Dataset classes

The signals look rather strange. May work better for beta band than alpha. Seems like it works better for smaller time
shifts
"""
import contextlib
import io
import os
import random

from scipy.signal import hilbert
import mne
import numpy
from matplotlib import pyplot

from cdl_eeg.data.paths import get_raw_data_storage_path
from cdl_eeg.models.transformations.amplitude_envelope import BivariateAmplitudeEnvelopeTimeShift
from cdl_eeg.models.transformations.utils import chunk_eeg, eeg_chunks_to_mne_epochs, UnivariateUniform


def main():
    # Make reproducible
    meaning_of_life = 42

    random.seed(meaning_of_life)
    numpy.random.seed(meaning_of_life)

    # -----------------
    # Hyperparameters
    # -----------------
    # Chunking
    num_chunks = 5
    chunk_duration = 2000
    chunk_time_delay = 1000
    ch_0, ch_1 = 0, 7

    # Permutation
    time_shift = UnivariateUniform(-100, 100)
    permute_first_channel = True

    # Data
    subject = 24
    derivatives = True  # boolean, indicating cleaned/not cleaned data

    # Preprocessing
    l_freq = 8
    h_freq = 12

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

    # Normalise
    data = (data - numpy.mean(data, axis=-1, keepdims=True)) / (numpy.std(data, axis=-1, keepdims=True) + 1e-8)

    # Chunk, both original and permuted
    original_chunks = chunk_eeg(data, k=num_chunks, chunk_duration=chunk_duration, delta_t=chunk_time_delay)

    transformation = BivariateAmplitudeEnvelopeTimeShift(time_shift=time_shift, num_chunks=num_chunks,
                                                         chunk_duration=chunk_duration,
                                                         chunk_time_delay=chunk_time_delay)
    permuted_chunks, idx, t0 = transformation.time_shift(x0=data[:, 0], x1=data[:, 1],
                                                         permute_first_channel=permute_first_channel)

    # -----------------
    # Convert to MNE objects, plot, and print details
    # -----------------
    # Need to remove batch dimensions
    original_chunks = tuple(numpy.squeeze(chunk, axis=0) for chunk in original_chunks)
    permuted_chunks = tuple(numpy.squeeze(chunk, axis=0) for chunk in permuted_chunks)

    # Create MNE objects
    original_epochs = eeg_chunks_to_mne_epochs(original_chunks, sampling_freq=eeg.info["sfreq"], verbose=verbose)
    permuted_epochs = eeg_chunks_to_mne_epochs(permuted_chunks, sampling_freq=eeg.info["sfreq"], verbose=verbose)

    # Print permutation details
    print("\n----- Permutation details -----")
    print(f"Permuted chunk (index): {idx}")
    print(f"Time shift t0: {t0 / eeg.info['sfreq']:.2f}s")

    # Plot
    # Amplitude envelopes
    lw = 2
    _, (ax1, ax2) = pyplot.subplots(2, 1)
    channel = 0 if permute_first_channel else 1

    # Get the amplitude
    original_amplitude = numpy.abs(hilbert(original_chunks[idx][channel]))
    time_shifted_amplitude = numpy.abs(hilbert(permuted_chunks[idx][channel]))

    ax1.plot(original_amplitude, label="Original Envelope", linewidth=lw)
    ax1.plot(original_chunks[idx][channel], label="Original Signal", linewidth=lw)

    ax2.plot(time_shifted_amplitude, label="Time Shifted Envelope", linewidth=lw)
    ax2.plot(permuted_chunks[idx][channel], label="Envelope Time Shifted Signal", linewidth=lw)

    ax1.legend()
    ax2.legend()

    if verbose:
        # MNE plots
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
