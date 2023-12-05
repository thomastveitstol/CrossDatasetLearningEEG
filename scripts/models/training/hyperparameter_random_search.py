import os
import random
import warnings

import yaml

from cdl_eeg.models.main_models.main_rbp_model import MainRBPModel


def main():
    # -----------------
    # Load settings (.yml file)
    # -----------------
    config_file = "hyperparameter_random_search.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_file)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------
    # Define MTS model
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

    # -----------------
    # Combine model with method for handling
    # a varied number of channels
    # todo: this needs to part of Random Search
    # -----------------
    # Filter some warnings from Voronoi split
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        model = MainRBPModel.from_config(rbp_config=config["RBP"], mts_config={"model": mts_module_name,
                                                                               "kwargs": mts_module_hyperparameters})


if __name__ == "__main__":
    main()
