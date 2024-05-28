"""
Script for plotting the power of different frequency bands, for all datasets
"""
import random
import warnings

import mne
import numpy
import seaborn
from matplotlib import pyplot
from mne.time_frequency import Spectrum
from scipy import integrate

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    # Make results reproducible
    random.seed(42)

    # Hyperparameters
    average_reference = True
    num_subjects_per_dataset = 200
    aggregation_method = "mean"
    z_normalise_all_channels = False

    power_bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}

    # Select datasets and define necessary keyword arguments for loading
    datasets = (HatlestadHall(), YulinWang(), Miltiadous())
    kwargs = {"HatlestadHall": {"derivatives": True, "session": "t1"},
              "YulinWang": {"derivatives": True, "visit": 1, "recording": "EC"},
              "Miltiadous": {}}

    # Loop through all datasets
    power_distributions = {"Dataset": [], "Frequency band": [], "Power": []}
    for dataset in datasets:
        dataset_name = type(dataset).__name__
        print(f"Computing power of dataset band: {dataset_name}")

        # Use a fixed number of subjects
        _all_subjects_available = dataset.get_subject_ids()
        subjects = random.sample(_all_subjects_available, k=min(num_subjects_per_dataset, len(_all_subjects_available)))

        # Store the excluded channels
        excluded_channels = kwargs[dataset_name].pop("excluded_channels", None)
        for subject in subjects:
            # --------------
            # Get and prepare the data
            # --------------
            # Get the MNE object
            with warnings.catch_warnings():  # todo: warnings exist for a reason, muting them is not that reason
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                raw = dataset.load_single_mne_object(subject_id=subject, **kwargs[dataset_name])

            # Maybe remove channels
            if excluded_channels is not None:
                if isinstance(excluded_channels, list):
                    raw = raw.pick(picks="eeg", exclude=excluded_channels)
                elif isinstance(excluded_channels, str):
                    raw = raw.pick(
                        picks="eeg",
                        exclude=[ch_name for ch_name in raw.ch_names if ch_name[:3] == excluded_channels]
                    )
                else:
                    raise TypeError(f"Excluded channels type {type(excluded_channels)} not understood")

            # Maybe re-reference to average
            if average_reference:
                raw.set_eeg_reference(ref_channels="average", verbose=False)

            # Maybe z-normalise all channels
            if z_normalise_all_channels:
                # Extract data as numpy array
                data = raw.get_data()

                # Subtract mean and divide by standard deviation
                data -= numpy.mean(data, keepdims=True, axis=-1)
                data /= numpy.std(data, keepdims=True, axis=-1)

                # Make it an MNE object
                raw = mne.io.RawArray(data=data, info=raw.info, verbose=False)

            # --------------
            # Compute power for all frequency bands
            # --------------
            # Compute the PSD
            psd: Spectrum = raw.compute_psd(verbose=False)

            # Loop through the frequency bands
            for frequency_band, (f_min, f_max) in power_bands.items():
                # Integrate between the desired range
                freqs = psd.freqs[(f_min < psd.freqs) & (psd.freqs < f_max)]
                psd_data = numpy.array(psd.get_data())[..., (f_min < psd.freqs) & (psd.freqs < f_max)]

                # The Simpson integration actually returns a numpy array, looks like scipy hasn't updated their type
                # hinting
                power: numpy.ndarray = integrate.simpson(y=psd_data, x=freqs, dx=None, axis=-1)  # type: ignore

                assert power.shape[0] == len(psd.ch_names), \
                    (f"Expected Simpson integration to give power per channel, but output dimension was "
                     f"{power.shape[0]}, while the number of channels is {len(psd.ch_names)}")

                # Add details
                power_distributions["Dataset"].append(dataset_name)
                power_distributions["Frequency band"].append(frequency_band)

                # Compute the average power across the channels and store it
                if aggregation_method == "mean":
                    agg = numpy.mean
                elif aggregation_method == "median":
                    agg = numpy.median
                else:
                    raise ValueError(f"Could not recognise aggregation method: {aggregation_method}")
                power_distributions["Power"].append(agg(power))

    # ---------------
    # Plotting
    # ---------------
    seaborn.violinplot(data=power_distributions, x="Frequency band", y="Power", hue="Dataset", density_norm="width")
    pyplot.title(f"Power distributions (N={num_subjects_per_dataset}), Aggregation method: "
                 f"{aggregation_method.capitalize()}, Channel normalisation: {z_normalise_all_channels}")

    pyplot.show()


if __name__ == "__main__":
    main()
