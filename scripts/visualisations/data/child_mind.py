from matplotlib import pyplot

from cdl_eeg.data.datasets.child_mind_dataset import ChildMind


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 13

    # Pre-processing
    filtering = None
    resampling_freq = None
    notch_filter = None
    avg_reference = False

    # -----------------
    # Load data
    # -----------------
    subject_id = ChildMind().get_subject_ids()[subject_number]
    eeg = ChildMind().load_single_mne_object(subject_id=subject_id)

    # Pre-process
    eeg = ChildMind().pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                                  resample=resampling_freq, excluded_channels=None, remove_above_std=None)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
