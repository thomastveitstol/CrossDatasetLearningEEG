"""
Script for visualising the latent features of a hidden layer using UMAP
"""
import os

import torch
import yaml

from cdl_eeg.data.paths import get_numpy_data_storage_path
from cdl_eeg.models.random_search.run_single_experiment import Experiment


def main():
    # ---------------
    # Load config files
    # ---------------
    path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Load file used for preprocessing the data
    preprocessed_path = os.path.join(get_numpy_data_storage_path(), config["preprocessed_version"],
                                     "preprocessing_config.yml")
    with open(preprocessed_path, "r") as f:
        preprocessed_config = yaml.safe_load(f)

    # ---------------
    # Run UMAP experiment
    # ---------------
    experiment = Experiment(config=config, pre_processing_config=preprocessed_config, results_path="None",
                            device=torch.device("cpu"))
    experiment.initial_hidden_layer_distributions()


if __name__ == "__main__":
    main()
