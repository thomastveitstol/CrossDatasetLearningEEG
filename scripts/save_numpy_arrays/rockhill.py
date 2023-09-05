from cdl_eeg.data.datasets.rockhill_dataset import Rockhill


def main():
    # ----------------
    # Hyperparameters  todo: currently running as testing/debugging only
    # ----------------
    # Pre-processing
    filtering = (1, 40)
    sampling_freq = 500
    notch_filter = None
    avg_reference = True

    # Cropping
    num_time_steps = sampling_freq * 16
    time_series_start = sampling_freq * 10

    # ----------------
    # Saving
    # ----------------
    Rockhill().save_eeg_as_numpy_arrays(subject_ids=None, filtering=filtering, resample=sampling_freq,
                                        notch_filter=notch_filter, avg_reference=avg_reference, derivatives=False,
                                        num_time_steps=num_time_steps, time_series_start=time_series_start, on=True)


if __name__ == "__main__":
    main()
