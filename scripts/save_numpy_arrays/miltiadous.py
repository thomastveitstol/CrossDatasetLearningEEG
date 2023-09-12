from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous


def main():
    # ----------------
    # Hyperparameters  todo: currently running as testing/debugging only
    # ----------------
    # Pre-processing
    derivatives = False  # setting derivatives to True may give boundary events
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
    Miltiadous().save_eeg_as_numpy_arrays(subject_ids=None, filtering=filtering, resample=sampling_freq,
                                          notch_filter=notch_filter, avg_reference=avg_reference,
                                          derivatives=derivatives, num_time_steps=num_time_steps,
                                          time_series_start=time_series_start)


if __name__ == "__main__":
    main()
