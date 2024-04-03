"""
Script for plotting the preprocessed version of the Miltiadous

Conclusion:
    Looks quite good, although not sure if all is resting state (see e.g. subject 56).
"""
from matplotlib import pyplot

from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 55
    derivatives = False

    # Pre-processing
    filtering = (1, 40)  # None  # (1, 40)
    resampling_freq = None
    notch_filter = None
    avg_reference = True
    remove_above_std = None  # 80e-06
    excluded_channels = None
    interpolation_method = "MNE"

    # -----------------
    # Load data
    # -----------------
    subject_id = Miltiadous().get_subject_ids()[subject_number]

    eeg = Miltiadous().load_single_mne_object(subject_id=subject_id, derivatives=derivatives)

    # Pre-process
    eeg = Miltiadous().pre_process(eeg, filtering=filtering, notch_filter=notch_filter,
                                   avg_reference=avg_reference, resample=resampling_freq,
                                   excluded_channels=excluded_channels, remove_above_std=remove_above_std,
                                   interpolation=interpolation_method)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
