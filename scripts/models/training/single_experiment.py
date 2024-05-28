"""
Script for running a single experiment, either leave-one-dataset-out cross validation, or its inverse
"""
import copy
import os
import random
from datetime import datetime, date

import yaml

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.models.random_search.generate_config_file import generate_config_file
from cdl_eeg.models.random_search.run_single_experiment import Experiment


def _str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError(f"Unexpected string: {s}")


def main():
    # ---------------
    # Load domains of design choices/hyperparameters (.yml file)
    # ---------------
    config_path = "hyperparameter_random_search.yml"
    path = os.path.join(os.path.dirname(__file__), "config_files", config_path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Create path and folder
    cv_method = random.choice(config["cv_method"])
    config["cv_method"] = cv_method
    if cv_method == "normal":
        _title_cv = "cv"
    elif cv_method == "inverted":
        _title_cv = "inverted_cv"
    else:
        raise ValueError
    results_path = os.path.join(get_results_dir(),
                                f"{config['selected_target']}_{_title_cv}_experiments_{date.today()}_"
                                f"{datetime.now().strftime('%H%M%S')}")
    os.mkdir(results_path)

    # ---------------
    # Generate config files
    # ---------------
    config, pre_processed_config = generate_config_file(config)

    # Save them
    with open(os.path.join(results_path, "config.yml"), "w") as file:
        yaml.safe_dump(config, file)

    with open(os.path.join(results_path, "preprocessing_config.yml"), "w") as file:
        yaml.safe_dump(pre_processed_config, file)

    # ---------------
    # Run experiment
    # ---------------
    # LODO or inverted LODO
    experiment_config = copy.deepcopy(config)
    if experiment_config["DomainDiscriminator"] is not None:
        num_train_datasets = len(experiment_config["Datasets"]) - 1
        experiment_config["DomainDiscriminator"]["discriminator"]["kwargs"]["num_classes"] = num_train_datasets

    print(f"\n{' Leave-one-dataset-out cross validation ':=^50}\n")
    with Experiment(config=experiment_config, pre_processing_config=pre_processed_config,
                    results_path=os.path.join(results_path, "leave_one_dataset_out")) as lodo_experiment:
        lodo_experiment.run_experiment()


if __name__ == "__main__":
    main()
