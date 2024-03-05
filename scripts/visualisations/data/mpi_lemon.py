from matplotlib import pyplot

from cdl_eeg.data.datasets.mpi_lemon import MPILemon


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 1
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

    eeg = MPILemon().load_single_mne_object(subject_id=subject_id, derivatives=derivatives,
                                            interpolation_method="spline")

    # Pre-process
    eeg = MPILemon().pre_process(eeg, filtering=filtering, notch_filter=notch_filter, avg_reference=avg_reference,
                                 resample=resampling_freq, excluded_channels=None, remove_above_std=None)

    # -----------------
    # Plot data
    # -----------------
    eeg.plot(verbose=False)
    eeg.compute_psd(verbose=False).plot()

    print(f"Number of channels: {len(eeg.ch_names)}")

    pyplot.show()


if __name__ == "__main__":
    main()
