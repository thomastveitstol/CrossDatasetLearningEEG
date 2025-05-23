import random

import numpy


def generate_preprocessing_config_file(config):
    # ---------------
    # Band pass filtering
    # ---------------
    # Select frequency band
    band = random.choice(config["BandPass"])

    # Sample from distribution
    if band["name"] in ("delta", "theta", "alpha", "beta", "gamma", "all"):
        l_freq = numpy.random.normal(loc=band["low"]["mean"], scale=band["low"]["std"])
        h_freq = numpy.random.normal(loc=band["high"]["mean"], scale=band["high"]["std"])
    else:
        raise ValueError(f"The band-pass name {band['name']} was not recognised")

    # ---------------
    # Other pre-processing hyperparameters
    # ---------------
    # Sampling frequency
    sampling_freq = h_freq * numpy.random.uniform(low=config["sampling_freq_high_freq_multiple"]["low"],
                                                  high=config["sampling_freq_high_freq_multiple"]["high"])

    # Number of time steps
    num_time_steps = int(sampling_freq * random.uniform(config["num_seconds"]["low"], config["num_seconds"]["high"]))

    # Number of time steps to skip
    time_series_start = config["time_series_start"]

    # Re-referencing
    avg_reference = config["avg_reference"]

    # Bad channels handling
    remove_above_std = random.uniform(config["remove_above_std"]["low"], config["remove_above_std"]["high"])
    interpolation = random.choice(config["interpolation"])

    # ---------------
    # Construct config dictionary (the dataset specifics are currently fixed)
    # ---------------
    general_config = {"filtering": (l_freq, h_freq), "resample": sampling_freq, "avg_reference": avg_reference,
                      "num_time_steps": num_time_steps, "time_series_start": time_series_start,
                      "remove_above_std": remove_above_std, "interpolation": interpolation}
    datasets = config["Datasets"]
    datasets["MPILemon"]["interpolation_method"] = interpolation

    return {"general": general_config, "datasets": datasets}
