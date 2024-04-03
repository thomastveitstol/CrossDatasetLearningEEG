"""
Script for plotting the preprocessed version of the Yulin Wang dataset, using tools from MNE.
"""
from matplotlib import pyplot

from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject = 57  # Number between (and including) 1 and 60
    recording = "EC"  # Can  be one of the following: 'EC', 'EO', 'Ma', 'Me', 'Mu'
    visit = 3  # Number between (and including) 1 and 3
    derivatives = True

    # Preprocessing
    l_freq = 1
    h_freq = 45
    notch_filter = None
    avg_reference = True
    resampling_freq = None
    excluded_channels = None
    remove_above_std = None

    # -----------------
    # Load data
    # -----------------
    subject_id = YulinWang().get_subject_ids()[subject]

    eeg = YulinWang().load_single_mne_object(subject_id=subject_id, derivatives=derivatives, visit=visit,
                                             recording=recording)

    # Pre-process
    eeg = YulinWang().pre_process(eeg, filtering=(l_freq, h_freq), notch_filter=notch_filter,
                                  avg_reference=avg_reference, resample=resampling_freq,
                                  excluded_channels=excluded_channels, remove_above_std=remove_above_std)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot(verbose=False)
    eeg.compute_psd(fmax=h_freq+15, verbose=False).plot()

    pyplot.show()


if __name__ == "__main__":
    main()
