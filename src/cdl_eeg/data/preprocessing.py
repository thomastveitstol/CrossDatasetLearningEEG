import os

import autoreject
import mne.io
import numpy
from matplotlib import pyplot


def create_folder_name(*, l_freq, h_freq, is_autorejected, resample_multiple):
    return f"data_band_pass_{l_freq}-{h_freq}_autoreject_{is_autorejected}_sampling_multiple_{resample_multiple}"


def _run_autoreject(epochs, autoreject_resample):
    if autoreject_resample is not None:
        epochs.resample(autoreject_resample, verbose=False)
    reject = autoreject.AutoReject(verbose=False)  # todo: hyperparameters
    return reject.fit_transform(epochs, return_log=True)


def _save_eeg_with_resampling_and_average_referencing(epochs: mne.Epochs, l_freq, h_freq, resample_fmax_multiples, path,
                                                      subject_id, is_autorejected, dataset_name: str, plot_data,
                                                      save_data):
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
            resampled_epochs.compute_psd().plot()
            pyplot.show()

        # Save numpy array
        if save_data:
            _folder_name = create_folder_name(l_freq=l_freq, h_freq=h_freq, is_autorejected=is_autorejected,
                                              resample_multiple=resample_multiple)
            array_path = os.path.join(path, _folder_name, dataset_name, f"{subject_id}.npy")
            numpy.save(array_path, arr=data)


def save_preprocessed_epochs(raw: mne.io.BaseRaw, *, excluded_channels, main_band_pass, frequency_bands, notch_filter,
                             num_epochs, epoch_duration, epoch_overlap, time_series_start_secs, autoreject_resample,
                             resample_fmax_multiples, subject_id, path, dataset_name, plot_data=False, save_data=True):
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

    assert num_epochs <= len(epochs), f"Cannot make {num_epochs} epochs when only {len(epochs)} are available"

    # Run autoreject
    autoreject_epochs, log = _run_autoreject(epochs.copy(), autoreject_resample=autoreject_resample)

    assert num_epochs <= len(autoreject_epochs), (f"Cannot make {num_epochs} epochs when only {len(autoreject_epochs)} "
                                                  f"are available after running autoreject")

    # Select epochs
    epochs = epochs[:num_epochs]
    autoreject_epochs = autoreject_epochs[:num_epochs]

    # Loop though and save EEG data for all frequency bands
    # todo: no need to loop through autoreject if nothing was changed
    for frequency_band in frequency_bands:
        l_freq, h_freq = frequency_band

        # Save non-autorejected
        _save_eeg_with_resampling_and_average_referencing(
            epochs=epochs.copy(), l_freq=l_freq, h_freq=h_freq, resample_fmax_multiples=resample_fmax_multiples,
            subject_id=subject_id, is_autorejected=False, path=path, plot_data=plot_data, dataset_name=dataset_name,
            save_data=save_data
        )

        # Save with autoreject
        _save_eeg_with_resampling_and_average_referencing(
            epochs=autoreject_epochs.copy(), l_freq=l_freq, h_freq=h_freq,
            resample_fmax_multiples=resample_fmax_multiples, subject_id=subject_id, is_autorejected=True, path=path,
            plot_data=plot_data, dataset_name=dataset_name, save_data=save_data
        )
