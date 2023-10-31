from matplotlib import pyplot

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 2
    derivatives = False

    # Pre-processing
    filtering = (1, 40)
    resampling_freq = None
    notch_filter = None
    avg_reference = False
    excluded_channels = ("Photic", "EKG")

    # -----------------
    # Load data
    # -----------------
    subject_id = CAUEEG().get_subject_ids()[subject_number]
    eeg = CAUEEG().load_single_mne_object(subject_id=subject_id, derivatives=derivatives)

    # Pre-process
    eeg = CAUEEG.pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                             resample=resampling_freq, excluded_channels=excluded_channels)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd().plot()
    print(eeg.ch_names)
    pyplot.show()


if __name__ == "__main__":
    main()
