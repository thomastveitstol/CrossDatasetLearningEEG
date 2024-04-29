import json
import os

import mne
from matplotlib import pyplot

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.preprocessing import save_preprocessed_epochs


def main():
    # -----------------
    # Hyperparameters
    # -----------------
    # Data
    subject_number = 33
    derivatives = False

    perform_preprocessing = False

    # Pre-processing
    excluded_channels = None
    main_band_pass = (1, 45)
    frequency_bands = ((1, 45),)
    notch_filter = None
    num_epochs = 5
    epoch_duration = 5
    epoch_overlap = 0
    time_series_start_secs = 30
    resample_fmax_multiples = (4, None)
    autoreject_resample = None
    seed = 1

    # -----------------
    # Load data
    # -----------------
    subject_id = CAUEEG().get_subject_ids()[subject_number]
    eeg = CAUEEG().load_single_mne_object(subject_id=subject_id, derivatives=derivatives)

    if perform_preprocessing:
        # Pre-process without saving
        save_preprocessed_epochs(
            eeg, excluded_channels=excluded_channels, main_band_pass=main_band_pass, frequency_bands=frequency_bands,
            notch_filter=notch_filter, num_epochs=num_epochs, epoch_duration=epoch_duration, epoch_overlap=epoch_overlap,
            time_series_start_secs=time_series_start_secs, resample_fmax_multiples=resample_fmax_multiples,
            autoreject_resample=autoreject_resample, dataset_name=None, path=None, subject_id=None, save_data=False,
            plot_data=True, seed=seed
        )
    else:
        path = os.path.join(CAUEEG().get_mne_path(), "event", f"{subject_id}.json")
        with open(path) as f:
            events = json.load(f)

        # ---------------
        # Create annotations object
        # ---------------
        # Get onsets and description
        onsets, descriptions = zip(*events)
        onsets = [single_onset / eeg.info["sfreq"] for single_onset in onsets]  # Converting to seconds

        # Guess the duration
        durations = [5] * len(onsets)

        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)

        eeg.set_annotations(annotations)
        eeg.plot()
        pyplot.show()


if __name__ == "__main__":
    main()
