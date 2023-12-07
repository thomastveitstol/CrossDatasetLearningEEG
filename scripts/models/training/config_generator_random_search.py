"""
This script generates a config yaml file, given the domains provided by a .yml file

The resulting config yml file will be used in both a k-fold cross validation script and a leave-one-dataset-out cross
validation script.
"""
import os
import random

import yaml


def main():
    # -----------------
    # Load domains of design choices/hyperparameters (.yml file)
    # -----------------
    config_file = "hyperparameter_random_search.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_file)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------
    # Sample training hyperparameters
    # -----------------
    train_hyperparameters = dict()
    for param, domain in config["Training"].items():
        if isinstance(domain, list):
            train_hyperparameters[param] = random.choice(domain)
        else:
            train_hyperparameters[param] = domain

    # -----------------
    # Sample method for handling a varied
    # number of channels
    # -----------------
    varied_numbers_of_channels = dict()
    method = random.choice(tuple(config["Varied Numbers of Channels"].keys()))
    if method == "RegionBasedPooling":
        # Todo: Implement how RBP should be sampled
        varied_numbers_of_channels[method] = config["Varied Numbers of Channels"][method]
    elif method == "SphericalSplineInterpolation":
        # Todo: Implement how to sample, and the code should NOT be in this script (even if it is trivial code)
        varied_numbers_of_channels[method] = random.choice(config["Varied Numbers of Channels"][method])
    else:
        raise ValueError(f"Expected method for handling varied numbers of EEG channels to be either region based "
                         f"pooling or spherical spline interpolation, but found {method}")

    # -----------------
    # Sample DL architecture and its hyperparameters
    # -----------------
    # Choose architecture
    mts_module_name = random.choice(tuple(config["MTS Module"].keys()))

    # Set hyperparameters
    mts_module_hyperparameters = dict()
    for hyperparameter_name, hyperparameter_domain in config["MTS Module"][mts_module_name].items():
        if isinstance(hyperparameter_domain, list):
            mts_module_hyperparameters[hyperparameter_name] = random.choice(hyperparameter_domain)
        else:
            mts_module_hyperparameters[hyperparameter_name] = hyperparameter_domain

    # Combine architecture name and hyperparameters in a dict
    dl_model = {"model": mts_module_name, "kwargs": mts_module_hyperparameters}

    # -----------------
    # Save as .yml file
    # -----------------
    safe_path = os.path.join(os.path.dirname(__file__), "random_search_config_files", "debug.yml")
    with open(safe_path, "w") as file:
        yaml.safe_dump(
            {"Training": train_hyperparameters, "Varied Numbers of Channels": varied_numbers_of_channels,
             "DL Architecture": dl_model},
            file)


if __name__ == "__main__":
    main()
