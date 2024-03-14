"""
Script for plotting the covariance matrices for all datasets
"""
import numpy
from matplotlib import pyplot

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    average_reference = True
    num_subjects_per_dataset = 50

    frequency_bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}

    datasets = (HatlestadHall(), YulinWang(), Miltiadous(), Rockhill(), CAUEEG())
    kwargs = {"HatlestadHall": {"derivatives": True, "session": "t1"},
              "YulinWang": {"derivatives": True, "visit": 1, "recording": "EC"},
              "Miltiadous": {},
              "Rockhill": {"on": True, "excluded_channels": "EXG"},
              "CAUEEG": {"excluded_channels": ["Photic", "EKG"]}}

    grand_covariance_matrices = {freq_band: dict() for freq_band in frequency_bands}
    for i, dataset in enumerate(datasets):
        dataset_name = type(dataset).__name__
        print(f"Computing covariance matrices for dataset {dataset_name} ({i+1}/{len(datasets)})")

        # Use a fixed number of subjects
        subjects = dataset.get_subject_ids()[:num_subjects_per_dataset]

        # Store the excluded channels
        excluded_channels = kwargs[dataset_name].pop("excluded_channels", None)

        covariance_matrices = {freq_band: [] for freq_band in frequency_bands}
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

            # Loop through the frequency bands
            for frequency_band, (f_min, f_max) in frequency_bands.items():
                filtered_raw = raw.copy()
                filtered_raw.filter(f_min, f_max, verbose=False)

                # --------------
                # Compute the covariance matrix, as in https://doi.org/10.1162/imag_a_00040
                # --------------
                x = filtered_raw.get_data()
                num_time_steps = x.shape[-1]

                covariance_matrices[frequency_band].append(
                    numpy.expand_dims(numpy.matmul(x, numpy.transpose(x)) / num_time_steps, axis=0)
                )

        # Compute grand covariance matrices
        for frequency_band, matrices in grand_covariance_matrices.items():
            matrices[dataset_name] = numpy.mean(numpy.concatenate(covariance_matrices[frequency_band], axis=0), axis=0)

    # --------------
    # Plotting and description
    # --------------
    decimals = 2
    for frequency_band, covariance_matrices in grand_covariance_matrices.items():
        print(f"\n{f' {frequency_band} ':=^35}")

        fig, axes = pyplot.subplots(1, len(datasets))
        for ax, (dataset_name, cov_matrix) in zip(axes, covariance_matrices.items()):
            # Descriptions
            print(f"\n{f' {dataset_name} ':-^25}")

            print(f"Mean log abs\t|{numpy.mean(numpy.log10(numpy.abs(cov_matrix))):8.{decimals}f}")
            print(f"Std log abs \t|{numpy.std(numpy.log10(numpy.abs(cov_matrix))):8.{decimals}f}")
            print(f"Max log abs \t|{numpy.max(numpy.log10(numpy.abs(cov_matrix))):8.{decimals}f}")
            print(f"Min log abs \t|{numpy.min(numpy.log10(numpy.abs(cov_matrix))):8.{decimals}f}")

            # Plotting
            ax.imshow(cov_matrix)
            ax.set_title(dataset_name)

        fig.suptitle(frequency_band)

    pyplot.show()


if __name__ == "__main__":
    main()
