"""
Script for plotting/developing a phase Cut/Paste transformation

(This is a script for developing. A PhaseCutPaste class inheriting from TransformationBase will be made in the future)

I must decide where to clip from. Some suggestions are: (1) a different segment from the same channel, (2) the same
segment from a different channel, (3) a different segment from a different channel, (4) a different subject (either same
or different channel. Can also be smart by selecting a subject here, by e.g. cutting and pasting from a demented
subject), (5) a synthetic phase.
"""
import contextlib
import io
import os

import mne
import numpy
from matplotlib import pyplot

from cdl_eeg.data.paths import get_raw_data_storage_path


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Permutation
    phase_shift = numpy.pi
    channel = 8

    # Data
    subject = 12
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

    # Apply the Hilbert transform to extract the analytic signal
    original_eeg = eeg.copy()
    eeg.apply_hilbert(envelope=False)  # envelope=False extracts the analytic signal

    # Get the phase information
    analytic_data = eeg.get_data()
    phase_data = numpy.angle(analytic_data[channel])  # Phase angle of the analytic signal in selected channel

    # Alter it by shifting
    phase_data += phase_shift

    modified_eeg_data = numpy.real(numpy.abs(analytic_data[channel]) * numpy.exp(1j * phase_data))

    # Create a permuted MNE object
    permuted_data = numpy.insert(numpy.real(eeg.get_data()[1:]), channel, numpy.expand_dims(modified_eeg_data, axis=0),
                                 axis=0)
    permuted_eeg = mne.io.RawArray(data=permuted_data, info=eeg.info, verbose=verbose)

    # Plot
    if verbose:
        original_eeg.plot(title="Original")
        permuted_eeg.plot(title="Permuted")

        pyplot.show()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            original_eeg.plot(title="Original")
            permuted_eeg.plot(title="Permuted")

            pyplot.show()


if __name__ == "__main__":
    main()
