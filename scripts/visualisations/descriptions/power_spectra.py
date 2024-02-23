"""
Script for plotting the power of different frequency bands, for all datasets
"""
from matplotlib import pyplot

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    average_reference = True
    num_subjects_per_dataset = 5

    datasets = (HatlestadHall(), YulinWang(), Miltiadous(), Rockhill(), CAUEEG())
    kwargs = {"HatlestadHall": {"derivatives": True, "session": "t1"},
              "YulinWang": {"derivatives": True, "visit": 1, "recording": "EC"},
              "Miltiadous": {},
              "Rockhill": {"on": True, "excluded_channels": "EXG"},
              "CAUEEG": {"excluded_channels": ["Photic", "EKG"]}}

    for dataset in datasets:
        dataset_name = type(dataset).__name__

        # Use a fixed number of subjects
        subjects = dataset.get_subject_ids()[:num_subjects_per_dataset]

        # Store the excluded channels
        excluded_channels = kwargs[dataset_name].pop("excluded_channels", None)
        for subject in subjects:
            # --------------
            # Get and prepare the data
            # --------------
            # Get the MNE object
            raw = dataset.load_single_mne_object(subject_id=subject, **kwargs[dataset_name])

            # Maybe remove channels
            if excluded_channels is not None:
                if isinstance(excluded_channels, list):
                    raw = raw.pick(picks="eeg", exclude=excluded_channels)
                elif isinstance(excluded_channels, str):
                    raw = raw.pick(
                        picks="eeg", exclude=[ch_name for ch_name in raw.ch_names if ch_name[:3] == excluded_channels]
                    )
                else:
                    raise TypeError(f"Excluded channels type {type(excluded_channels)} not understood")

            # Maybe re-reference to average
            if average_reference:
                raw.set_eeg_reference(ref_channels="average", verbose=False)

            # --------------
            # Compute power
            # --------------
            # Compute the PSD
            raw.compute_psd(fmin=0, fmax=45, verbose=False).plot()

            pyplot.show()


if __name__ == "__main__":
    main()
