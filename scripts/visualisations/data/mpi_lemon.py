from matplotlib import pyplot

from cdl_eeg.data.datasets.mpi_lemon import MPILemon


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 90
    derivatives = False

    # Pre-processing
    filtering = (1, 40)
    resampling_freq = None
    notch_filter = None
    avg_reference = True

    # -----------------
    # Load data
    # -----------------
    subject_id = MPILemon().get_subject_ids()[subject_number]
    eeg = MPILemon().load_single_mne_object(subject_id=subject_id, derivatives=derivatives)

    # Pre-process
    eeg = MPILemon.pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                               resample=resampling_freq)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot()
    eeg.compute_psd().plot()

    pyplot.show()


if __name__ == "__main__":
    main()
