"""
CAUEEG
"""
import random

import mne
import numpy
import seaborn
from matplotlib import pyplot
from mne.time_frequency import Spectrum
from scipy import integrate

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG


def _compute_band_power(psd, f_min, f_max, aggregation_method):
    # Integrate between the desired range
    freqs = psd.freqs[(f_min < psd.freqs) & (psd.freqs < f_max)]
    psd_data = numpy.log(numpy.array(psd.get_data())[..., (f_min < psd.freqs) & (psd.freqs < f_max)])

    # The Simpson integration actually returns a numpy array, looks like scipy hasn't updated their type
    # hinting
    power: numpy.ndarray = integrate.simpson(y=psd_data, x=freqs, dx=None, axis=-1)  # type: ignore

    assert power.shape[0] == len(psd.ch_names), \
        (f"Expected Simpson integration to give power per channel, but output dimension was "
         f"{power.shape[0]}, while the number of channels is {len(psd.ch_names)}")

    # Compute the average power across the channels and store it
    if aggregation_method == "mean":
        agg = numpy.mean
    elif aggregation_method == "median":
        agg = numpy.median
    else:
        raise ValueError(f"Could not recognise aggregation method: {aggregation_method}")

    # Return power
    return agg(power)


def main():
    # Make results reproducible
    random.seed(42)

    # Hyperparameters
    band_1 = "Theta"  # Numerator
    band_2 = "Alpha"  # Denominator

    tmin, tmax = 5, 55
    average_reference = True
    num_subjects_per_dataset = 200
    aggregation_method = "mean"
    z_normalise_all_channels = False

    power_bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}

    # Select datasets and define necessary keyword arguments for loading
    dataset = CAUEEG()
    kwargs = {"HatlestadHall": {"derivatives": True, "session": "t1"},
              "YulinWang": {"derivatives": True, "visit": 1, "recording": "EC"},
              "Miltiadous": {},
              "CAUEEG": {"excluded_channels": ["Photic", "EKG"]},
              "MPILemon": {"interpolation_method": "MNE"}}

    # Some printing an
    dataset_name = type(dataset).__name__
    print(f"Computing power of dataset band: {dataset_name}")

    # Use a fixed number of subjects
    _all_subjects_available = dataset.get_subject_ids()
    subjects = random.sample(_all_subjects_available, k=min(num_subjects_per_dataset, len(_all_subjects_available)))

    # Store the excluded channels
    excluded_channels = kwargs[dataset_name].pop("excluded_channels", None)
    details = {f"{band_1}/{band_2}": [], band_1: [], band_2: [], "cognition": [], "covariances": []}
    for subject in subjects:
        # Cognition
        mci = dataset.mci((subject,))[0]
        ad = dataset.alzheimers((subject,))[0]
        normal = dataset.normal((subject,))[0]
        if not any((mci, ad, normal)):
            continue
        elif sum((mci, ad, normal)) != 1:
            raise ValueError(f"Expected only one condition to be fulfilled, but found {(mci, ad, normal)=}")
        else:
            if mci:
                condition = "mci"
            elif ad:
                condition = "ad"
            elif normal:
                condition = "normal"
            else:
                raise ValueError("This should never happen...")

        # --------------
        # Get and prepare the data
        # --------------
        # Get the MNE object
        raw = dataset.load_single_mne_object(subject_id=subject, **kwargs[dataset_name])

        # Crop
        raw.crop(tmin, tmax)

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

        # Get the band-powers
        power_1 = _compute_band_power(psd, f_min=power_bands[band_1][0], f_max=power_bands[band_1][1],
                                      aggregation_method=aggregation_method)
        power_2 = _compute_band_power(psd, f_min=power_bands[band_2][0], f_max=power_bands[band_2][1],
                                      aggregation_method=aggregation_method)

        # Covariance
        _x = raw.get_data()
        cov = numpy.mean(numpy.abs(numpy.matmul(_x, _x.T) / _x.shape[-1]))

        # Store
        details[f"{band_1}/{band_2}"].append(power_1 / power_2)
        details[band_1].append(power_1)
        details[band_2].append(power_2)
        details["covariances"].append(cov)
        details["cognition"].append(condition)

    # ---------------
    # Plotting
    # ---------------
    for x_var in details:
        if x_var == "cognition":
            continue

        pyplot.figure()
        seaborn.violinplot(data=details, x=x_var, y="cognition")
        pyplot.title(f"Distributions (N={num_subjects_per_dataset})")

    pyplot.show()


if __name__ == "__main__":
    main()
