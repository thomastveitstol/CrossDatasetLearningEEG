import random

import numpy


def generate_preprocessing_config_file(config):
    # ---------------
    # Band pass filtering
    # ---------------
    # Select frequency band
    band = random.choice(config["BandPass"])

    # Sample from distribution
    if band["name"] in ("delta", "theta", "alpha", "beta", "gamma"):
        l_freq = numpy.random.normal(loc=band["low"]["mean"], scale=band["low"]["std"])
        h_freq = numpy.random.normal(loc=band["high"]["mean"], scale=band["high"]["std"])
    elif band["name"] == "uniform":
        l_freq = numpy.random.uniform(low=band["low"]["min"], high=band["low"]["max"])
        h_freq = numpy.random.uniform(low=max(band["high"]["min"], l_freq), high=band["high"]["max"])
    else:
        raise ValueError(f"The band-pass name {band['name']} was not recognised")

    # ---------------
    # Other pre-processing hyperparameters
    # ---------------
    # Sampling frequency
    sampling_freq = h_freq * numpy.random.uniform(low=config["sampling_freq_high_freq_multiple"]["low"],
                                                  high=config["sampling_freq_high_freq_multiple"]["high"])

    # Number of time steps
    num_time_steps = int(sampling_freq * random.choice(config["num_seconds"]))

    # Number of time steps to skip
    time_series_start = int(sampling_freq * random.choice(config["time_series_start"]))

    # Re-referencing
    avg_reference = random.choice(config["avg_reference"])

    # Using already pre-processed version
    # derivatives = random.choice(config["derivatives"])

    # ---------------
    # Construct config dictionary (the dataset specifics are currently fixed)
    # ---------------
    general_config = {"filtering": (l_freq, h_freq), "resample": sampling_freq, "avg_reference": avg_reference,
                      "num_time_steps": num_time_steps, "time_series_start": time_series_start}
    return {"general": general_config, "datasets": config["Datasets"]}
