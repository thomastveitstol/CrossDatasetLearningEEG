"""
Script for plotting the results of the phase replacement transform

This transformation looks a little strange. Also, the frequency is shifted when replacing the phase
"""
import contextlib
import io
import os
import random

import mne
import numpy
from matplotlib import pyplot


from cdl_eeg.data.paths import get_raw_data_storage_path
from cdl_eeg.models.transformations.phase_consistency import PhaseReplacement
from cdl_eeg.models.transformations.utils import chunk_eeg, eeg_chunks_to_mne_epochs


def main():
    random.seed(1)
    numpy.random.seed(2)

    # -----------------
    # Hyperparameters
    # -----------------
    # Chunking
    num_chunks = 5
    chunk_duration = 2000
    chunk_time_delay = 1000

    # Data. The phase of subject 1 will be inserted into subject 0
    subject_0 = 6
    subject_1 = 24

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

    # Subject 0
    subject_id_0 = tuple(dataset for dataset in os.listdir(dataset_path) if dataset[:4] == "sub-")[subject_0]
    path_0 = os.path.join(dataset_path, subject_id_0, "eeg", f"{subject_id_0}_task-eyesclosed_eeg.set")

    eeg_0: mne.io.eeglab.eeglab.RawEEGLAB = mne.io.read_raw_eeglab(input_fname=path_0, preload=True, verbose=verbose)

    # Subject 1
    subject_id_1 = tuple(dataset for dataset in os.listdir(dataset_path) if dataset[:4] == "sub-")[subject_1]
    path_1 = os.path.join(dataset_path, subject_id_1, "eeg", f"{subject_id_1}_task-eyesclosed_eeg.set")

    eeg_1: mne.io.eeglab.eeglab.RawEEGLAB = mne.io.read_raw_eeglab(input_fname=path_1, preload=True, verbose=verbose)

    # -----------------
    # Some additional preprocessing steps
    # -----------------
    # Filtering
    eeg_0.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)
    eeg_1.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # Re-referencing
    eeg_0.set_eeg_reference(ref_channels="average", verbose=verbose)
    eeg_1.set_eeg_reference(ref_channels="average", verbose=verbose)

    # -----------------
    # Split into chunks
    # -----------------
    # Convert to numpy arrays and add a batch dimension
    data_0 = numpy.expand_dims(eeg_0.get_data(), axis=0)
    data_1 = numpy.expand_dims(eeg_1.get_data(), axis=0)

    # Normalise
    data_0 = (data_0 - numpy.mean(data_0, axis=-1, keepdims=True)) / (numpy.std(data_0, axis=-1, keepdims=True) + 1e-8)
    data_1 = (data_1 - numpy.mean(data_1, axis=-1, keepdims=True)) / (numpy.std(data_1, axis=-1, keepdims=True) + 1e-8)

    # Chunk, both original and permuted
    chunks_0 = chunk_eeg(data_0, k=num_chunks, chunk_duration=chunk_duration, delta_t=chunk_time_delay)
    chunks_1 = chunk_eeg(data_1, k=num_chunks, chunk_duration=chunk_duration, delta_t=chunk_time_delay)

    transformation = PhaseReplacement(num_chunks=num_chunks, chunk_duration=chunk_duration,
                                      chunk_time_delay=chunk_time_delay)
    permuted_chunks, idx = transformation.phase_replacement(original_data=data_0, replacement_data=data_1)
    # -----------------
    # Convert to MNE objects, plot, and print details
    # -----------------
    # Need to remove batch dimensions
    chunks_0 = tuple(numpy.squeeze(chunk, axis=0) for chunk in chunks_0)
    chunks_1 = tuple(numpy.squeeze(chunk, axis=0) for chunk in chunks_1)
    permuted_chunks = tuple(numpy.squeeze(chunk, axis=0) for chunk in permuted_chunks)

    # Create MNE objects
    epochs_0 = eeg_chunks_to_mne_epochs(chunks_0, sampling_freq=eeg_0.info["sfreq"], verbose=verbose)
    epochs_1 = eeg_chunks_to_mne_epochs(chunks_1, sampling_freq=eeg_1.info["sfreq"], verbose=verbose)

    permuted_epochs = eeg_chunks_to_mne_epochs(permuted_chunks, sampling_freq=eeg_0.info["sfreq"], verbose=verbose)

    # Print permutation details
    print("\n----- Permutation details -----")
    print(f"Permuted chunk (index): {idx}\n")

    # Plot
    if verbose:
        epochs_0.plot(picks="all", title="Subject 0", events=False, scalings=3)
        epochs_1.plot(picks="all", title="Subject 1", events=False, scalings=3)

        permuted_epochs.plot(picks="all", title="Permuted", events=False, scalings=3)

        pyplot.show()
    else:
        # Redirect to an unused StringIO object
        with contextlib.redirect_stdout(io.StringIO()):
            epochs_0.plot(picks="all", title="Subject 0", events=False, scalings=3)
            epochs_1.plot(picks="all", title="Subject 1", events=False, scalings=3)

            permuted_epochs.plot(picks="all", title="Permuted", events=False, scalings=3)

            pyplot.show()


if __name__ == "__main__":
    main()
