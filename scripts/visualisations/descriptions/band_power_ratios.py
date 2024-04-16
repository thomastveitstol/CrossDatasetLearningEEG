"""
Subjects 004 and 103 have theta / alpha > 2
"""
import random

import mne
import numpy
from matplotlib import pyplot
from mne.time_frequency import Spectrum
from scipy import integrate
from sklearn.linear_model import LinearRegression

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.mpi_lemon import MPILemon


def _compute_band_power(psd, f_min, f_max, aggregation_method):
    # Integrate between the desired range
    freqs = psd.freqs[(f_min < psd.freqs) & (psd.freqs < f_max)]
    psd_data = numpy.array(psd.get_data())[..., (f_min < psd.freqs) & (psd.freqs < f_max)]

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
    band_1 = "Beta"  # Numerator
    band_2 = "Delta"  # Denominator

    average_reference = True
    num_subjects_per_dataset = 50
    aggregation_method = "mean"
    z_normalise_all_channels = False

    power_bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30), "Gamma": (30, 45)}

    # Select datasets and define necessary keyword arguments for loading
    dataset = HatlestadHall()
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
    band_power_ratios = []
    ages = []
    powers_1 = []
    powers_2 = []
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

        # Get the age
        age = dataset.age((subject,))[0]

        # Store
        band_power_ratios.append(power_1 / power_2)
        ages.append(age)
        powers_1.append(power_1)
        powers_2.append(power_2)

        if power_1 / power_2 > 2:
            print(subject)
            # raw.plot()
            # psd.plot()
            # pyplot.show()

    # ---------------
    # Linear regression
    # ---------------
    lin_reg_ratio = LinearRegression()
    lin_reg_power_1 = LinearRegression()
    lin_reg_power_2 = LinearRegression()

    lin_reg_ratio.fit(numpy.expand_dims(ages, axis=-1), numpy.expand_dims(band_power_ratios, axis=-1))
    lin_reg_power_1.fit(numpy.expand_dims(ages, axis=-1), numpy.expand_dims(powers_1, axis=-1))
    lin_reg_power_2.fit(numpy.expand_dims(ages, axis=-1), numpy.expand_dims(powers_2, axis=-1))

    # ---------------
    # Plotting
    # ---------------
    for lin_reg, features, y_label in zip((lin_reg_ratio, lin_reg_power_1, lin_reg_power_2),
                                          (band_power_ratios, powers_1, powers_2),
                                          (f"{band_1} / {band_2}", band_1, band_2)):
        pyplot.figure()

        # Plot actual scores
        pyplot.plot(ages, features, ".")

        # Plot linear fit
        _x0 = min(ages)
        _x1 = max(ages)

        _y0 = lin_reg.predict(numpy.expand_dims([_x0], axis=-1))[0]
        _y1 = lin_reg.predict(numpy.expand_dims([_x1], axis=-1))[0]
        pyplot.plot([_x0, _x1], [_y0, _y1])

        pyplot.xlabel("Age")
        pyplot.ylabel(y_label)
        _score = lin_reg.score(numpy.expand_dims(ages, axis=-1), numpy.expand_dims(features, axis=-1))
        pyplot.title(f"R^2: {_score}")

    pyplot.show()


if __name__ == "__main__":
    main()
