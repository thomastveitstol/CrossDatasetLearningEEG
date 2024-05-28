"""
Functions for pre-processing and saving data as numpy arrays
"""
import os

import autoreject
import mne.io
import numpy
from matplotlib import pyplot


# -------------
# Helpful functions
# -------------
def create_folder_name(*, l_freq, h_freq, is_autorejected, resample_multiple):
    """Function for creating a string to be used as a folder name"""
    return f"data_band_pass_{l_freq}-{h_freq}_autoreject_{is_autorejected}_sampling_multiple_{resample_multiple}"


def _run_autoreject(epochs, autoreject_resample, seed, num_splits):
    """Function for running autoreject"""
    if autoreject_resample is not None:
        epochs.resample(autoreject_resample, verbose=False)
    reject = autoreject.AutoReject(verbose=False, random_state=seed, cv=num_splits)
    return reject.fit_transform(epochs, return_log=True)


def _save_eeg_with_resampling_and_average_referencing(epochs: mne.Epochs, l_freq, h_freq, resample_fmax_multiples, path,
                                                      subject_id, is_autorejected, dataset_name: str, plot_data,
                                                      save_data):
    """Function for saving EEG data as numpy arrays, which has already been pre-processed to some extent"""
    # Perform band-pass filtering
    epochs.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # Loop through all resampling frequencies
    for resample_multiple in resample_fmax_multiples:
        resampled_epochs = epochs.copy()

        if resample_multiple is not None:
            # Calculate the new frequency
            new_freq = resample_multiple * h_freq

            # Perform resampling
            resampled_epochs.resample(new_freq, verbose=False)

        # Re-reference to average
        resampled_epochs.set_eeg_reference(ref_channels="average", verbose=False)

        # Convert to numpy arrays
        data = resampled_epochs.get_data()
        assert data.ndim == 3, (f"Expected the EEG data to have three dimensions (epochs, channels, time steps), but "
                                f"found shape={data.shape}")

        # Maybe plot the data
        if plot_data:
            print("--------------------------")
            print(f"Band-pass filter: {l_freq, h_freq}")
            print(f"Sampling rate: f_max * {resample_multiple}")
            print(f"Autorejected: {is_autorejected}")
            resampled_epochs.plot(scalings="auto")
            resampled_epochs.compute_psd(verbose=False).plot()
            pyplot.show()

        # Save numpy array
        if save_data:
            _folder_name = create_folder_name(l_freq=l_freq, h_freq=h_freq, is_autorejected=is_autorejected,
                                              resample_multiple=resample_multiple)
            array_path = os.path.join(path, _folder_name, dataset_name, f"{subject_id}.npy")
            numpy.save(array_path, arr=data)


# -------------
# Main saving function
# -------------
def save_preprocessed_epochs(raw: mne.io.BaseRaw, *, excluded_channels, main_band_pass, frequency_bands, notch_filter,
                             num_epochs, epoch_duration, epoch_overlap, time_series_start_secs, autoreject_resample,
                             resample_fmax_multiples, subject_id, path, dataset_name, seed, plot_data=False,
                             save_data=True):
    """Main function for saving a range of different pre-processed versions of the same EEG data"""
    # ---------------
    # Input checks
    # ---------------
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"Unexpected type: {type(raw)}")

    # ---------------
    # Pre-processing steps
    # ---------------
    # Remove channels
    if excluded_channels is not None:
        raw = raw.pick(picks="eeg", exclude=excluded_channels)

    # Crop
    if time_series_start_secs is not None:
        raw.crop(tmin=time_series_start_secs)

    # Band-pass filtering
    if main_band_pass is not None:
        raw.filter(*main_band_pass, verbose=False)

    # Notch filter
    if notch_filter is not None:
        raw.notch_filter(notch_filter, verbose=False)

    # Epoch the data
    epochs: mne.Epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True, overlap=epoch_overlap,
                                                      verbose=False)

    if num_epochs > len(epochs):
        # Skipping subject if insufficient number of epochs
        return None

    # Run autoreject
    autoreject_epochs, log = _run_autoreject(epochs.copy(), autoreject_resample=autoreject_resample, seed=seed,
                                             num_splits=min(10, len(epochs)))

    # Select epochs
    epochs = epochs[:num_epochs]
    if num_epochs > len(autoreject_epochs):
        # Skipping autorejected epochs if insufficient amount
        autoreject_epochs = None
    else:
        autoreject_epochs = autoreject_epochs[:num_epochs]

    # Loop though and save EEG data for all frequency bands
    for frequency_band in frequency_bands:
        l_freq, h_freq = frequency_band

        # Save non-autorejected
        _save_eeg_with_resampling_and_average_referencing(
            epochs=epochs.copy(), l_freq=l_freq, h_freq=h_freq, resample_fmax_multiples=resample_fmax_multiples,
            subject_id=subject_id, is_autorejected=False, path=path, plot_data=plot_data, dataset_name=dataset_name,
            save_data=save_data
        )

        # (Maybe) save with autoreject
        if autoreject_epochs is not None:
            _save_eeg_with_resampling_and_average_referencing(
                epochs=autoreject_epochs.copy(), l_freq=l_freq, h_freq=h_freq,
                resample_fmax_multiples=resample_fmax_multiples, subject_id=subject_id, is_autorejected=True, path=path,
                plot_data=plot_data, dataset_name=dataset_name, save_data=save_data
            )
