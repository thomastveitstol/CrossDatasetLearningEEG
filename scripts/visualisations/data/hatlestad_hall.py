import mne
from matplotlib import pyplot

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 9
    derivatives = True
    session = "t1"

    # Pre-processing
    filtering = None
    resampling_freq = None
    notch_filter = None
    avg_reference = False

    # -----------------
    # Load data
    # -----------------
    subject_id = HatlestadHall().get_subject_ids()[subject_number]
    eeg = HatlestadHall().load_single_mne_object(subject_id=subject_id, derivatives=derivatives, session=session)

    # Pre-process
    eeg = HatlestadHall().pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                                      resample=resampling_freq, remove_above_std=None, excluded_channels=None)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot(noise_cov=mne.compute_raw_covariance(raw=eeg, verbose=False))
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
