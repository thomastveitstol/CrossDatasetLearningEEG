from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall


def main():
    # ----------------
    # Hyperparameters  todo: currently running as testing/debugging only
    # ----------------
    # Data
    session = "t1"

    # Pre-processing
    derivatives = True
    filtering = (1, 40)
    sampling_freq = 500
    notch_filter = None
    avg_reference = False  # Not needed, as already done in the pipeline

    # Cropping
    num_time_steps = sampling_freq * 25
    time_series_start = sampling_freq * 30

    # ----------------
    # Saving
    # ----------------
    HatlestadHall().save_eeg_as_numpy_arrays(subject_ids=None, filtering=filtering, resample=sampling_freq,
                                             notch_filter=notch_filter, avg_reference=avg_reference,
                                             derivatives=derivatives, num_time_steps=num_time_steps,
                                             time_series_start=time_series_start, session=session)


if __name__ == "__main__":
    main()
