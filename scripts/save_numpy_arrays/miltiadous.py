from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous


def main():
    # ----------------
    # Hyperparameters  todo: currently running as testing/debugging only
    # ----------------
    # Pre-processing
    filtering = (1, 40)
    resample = 500
    notch_filter = 50
    avg_reference = True

    # Cropping
    num_time_steps = resample * 16
    time_series_start = resample * 10

    # ----------------
    # Saving
    # ----------------
    Miltiadous().save_eeg_as_numpy_arrays(subject_ids=None, filtering=filtering, resample=resample,
                                          notch_filter=notch_filter, avg_reference=avg_reference,
                                          num_time_steps=num_time_steps, time_series_start=time_series_start)


if __name__ == "__main__":
    main()
