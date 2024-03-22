from matplotlib import pyplot

from cdl_eeg.data.datasets.van_hees import VanHees


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_id = "signal-6-1"

    # Pre-processing
    filtering = (1, 40)
    resampling_freq = None
    notch_filter = None
    avg_reference = True

    # -----------------
    # Load data
    # -----------------
    eeg = VanHees().load_single_mne_object(subject_id=subject_id)

    # Pre-process
    eeg = VanHees().pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                                resample=resampling_freq, excluded_channels=None, remove_above_std=None)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
