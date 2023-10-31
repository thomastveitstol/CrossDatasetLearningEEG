from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG


def main():
    # ----------------
    # Hyperparameters
    # ----------------
    # Pre-processing
    derivatives = False
    filtering = (1, 40)
    sampling_freq = 500
    notch_filter = None
    avg_reference = True
    non_eeg_channels = ("Photic", "EKG")

    # Cropping
    num_time_steps = sampling_freq * 25
    time_series_start = sampling_freq * 30

    # ----------------
    # Saving
    # ----------------
    CAUEEG().save_eeg_as_numpy_arrays(subject_ids=None, filtering=filtering, resample=sampling_freq,
                                      notch_filter=notch_filter, avg_reference=avg_reference, derivatives=derivatives,
                                      num_time_steps=num_time_steps, time_series_start=time_series_start,
                                      excluded_channels=non_eeg_channels)


if __name__ == "__main__":
    main()
