"""

"""
from matplotlib import pyplot

from cdl_eeg.data.datasets.maruyama_dataset import Maruyama


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 1

    # Pre-processing
    filtering = (1, 40)
    resampling_freq = None
    notch_filter = 50
    avg_reference = False

    # -----------------
    # Load data
    # -----------------
    subject_id = Maruyama().get_subject_ids()[subject_number]
    eeg = Maruyama().load_single_mne_object(subject_id=subject_id, derivatives=False)
    print(eeg)

    # Pre-process
    eeg = Maruyama.pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                               resample=resampling_freq)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
