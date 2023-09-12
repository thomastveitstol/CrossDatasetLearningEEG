from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    # ----------------
    # Hyperparameters  todo: currently running as testing/debugging only
    # ----------------
    # Data
    visit = 1
    recording = "EC"

    # Pre-processing
    derivatives = True
    filtering = (1, 40)
    sampling_freq = 500
    notch_filter = 50
    avg_reference = True

    # Cropping
    num_time_steps = sampling_freq * 25
    time_series_start = sampling_freq * 30

    # ----------------
    # Saving
    # ----------------
    YulinWang().save_eeg_as_numpy_arrays(subject_ids=None, filtering=filtering, resample=sampling_freq,
                                         notch_filter=notch_filter, avg_reference=avg_reference,
                                         num_time_steps=num_time_steps, time_series_start=time_series_start,
                                         derivatives=derivatives, visit=visit, recording=recording)


if __name__ == "__main__":
    main()
