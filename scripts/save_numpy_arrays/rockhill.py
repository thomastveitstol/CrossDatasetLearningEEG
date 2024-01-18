from cdl_eeg.data.datasets.rockhill_dataset import Rockhill


def main():
    # ----------------
    # Hyperparameters  todo: currently running as testing/debugging only
    # ----------------
    # Pre-processing
    filtering = (1, 45)
    sampling_freq = 180
    notch_filter = None
    avg_reference = True

    # Cropping
    num_time_steps = sampling_freq * 25
    time_series_start = sampling_freq * 30

    # ----------------
    # Saving
    # ----------------
    Rockhill().save_eeg_as_numpy_arrays(subject_ids=None, filtering=filtering, resample=sampling_freq,
                                        notch_filter=notch_filter, avg_reference=avg_reference, derivatives=False,
                                        num_time_steps=num_time_steps, time_series_start=time_series_start,
                                        on=True)  # TODO!: remove the 'on' argument


if __name__ == "__main__":
    main()
