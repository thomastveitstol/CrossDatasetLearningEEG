import contextlib
import io
import os
import mne
import numpy
from matplotlib import pyplot

from cdl_eeg.data.paths import get_raw_data_storage_path
from cdl_eeg.models.transformations.frequency_slowing import FrequencySlowing
from cdl_eeg.models.transformations.utils import UnivariateNormal


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Permutation
    slowing_distribution = UnivariateNormal(0.7, 0.1)

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
    # Get data and perform permutation
    # -----------------
    data = eeg.get_data()

    # Normalise
    data = (data - numpy.mean(data, axis=-1, keepdims=True)) / (numpy.std(data, axis=-1, keepdims=True) + 1e-8)

    # Transformation
    transformation = FrequencySlowing(slowing_distribution=slowing_distribution)
    permuted_data, phase_modulation = transformation.phase_slowing(data)

    # -----------------
    # Convert to MNE objects, plot, and print details
    # -----------------
    # Create MNE objects. Need to re-make the info object
    original_raw = mne.io.RawArray(data, mne.create_info(sfreq=eeg.info["sfreq"], ch_names=eeg.info["ch_names"]),
                                   verbose=verbose)
    permuted_raw = mne.io.RawArray(permuted_data,
                                   mne.create_info(sfreq=eeg.info["sfreq"], ch_names=eeg.info["ch_names"]),
                                   verbose=verbose)

    # Print permutation details
    print("\n----- Permutation details -----")
    print(f"Phase modulation: {phase_modulation:.3f}")

    # Plot
    if verbose:
        # MNE plots
        original_raw.plot(title="Original", scalings=3)
        permuted_raw.plot(title="Permuted", scalings=3)

        original_raw.compute_psd(title="Original").plot()
        permuted_raw.compute_psd(title="Permuted").plot()

        pyplot.show()
    else:
        # Redirect to an unused StringIO object
        with contextlib.redirect_stdout(io.StringIO()):
            original_raw.plot(title="Original", scalings=3)
            permuted_raw.plot(title="Permuted", scalings=3)

            original_raw.compute_psd(picks="all").plot(picks="all")
            permuted_raw.compute_psd(picks="all").plot(picks="all")

            pyplot.show()


if __name__ == "__main__":
    main()
