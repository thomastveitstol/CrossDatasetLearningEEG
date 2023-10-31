from matplotlib import pyplot

from cdl_eeg.data.datasets.ous_dataset import OUS


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 9
    derivatives = False

    # Pre-processing
    filtering = None
    resampling_freq = None
    notch_filter = None
    avg_reference = False

    # -----------------
    # Load data
    # -----------------
    subject_id = OUS().get_subject_ids()[subject_number]
    eeg = OUS().load_single_mne_object(subject_id=subject_id, derivatives=derivatives)

    # Pre-process
    eeg = OUS.pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                          resample=resampling_freq)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
