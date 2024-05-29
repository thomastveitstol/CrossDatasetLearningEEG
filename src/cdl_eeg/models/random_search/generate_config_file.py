import os
import random
from typing import Any, Dict, Optional

import yaml  # type: ignore[import]

from cdl_eeg.data.paths import get_numpy_data_storage_path
from cdl_eeg.models.domain_adaptation.domain_discriminators.getter import get_domain_discriminator_type
from cdl_eeg.models.mts_modules.getter import get_mts_module_type
from cdl_eeg.models.region_based_pooling.hyperparameter_sampling import sample_rbp_designs
from cdl_eeg.models.random_search.sampling_distributions import sample_hyperparameter


def _str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError(f"Unexpected string: {s}")


def generate_config_file(config):
    """
    Function for generating a config file (.yml format)

    Parameters
    ----------
    config : dict[str, typing.Any]
        The config file containing info regarding the domains of which to sample the hyperparameters/design choices from

    Returns
    -------
    dict[str, typing.Any]
    """
    # -----------------
    # Add details for splitting in folds
    # -----------------
    subjects_split_hyperparameters = config["SubjectSplit"]

    # -----------------
    # Add details for computing metrics on sub-groups
    # -----------------
    sub_groups_hyperparameters = config["SubGroups"]

    # -----------------
    # Add dataset details
    # -----------------
    dataset_hyperparameters = config["Datasets"]

    # Delete datasets which are incompatible with the desired target  todo: could check the class instead
    datasets_to_delete = tuple(dataset_name for dataset_name, hyperparams in dataset_hyperparameters.items()
                               if config["Training"]["target"] not in hyperparams["target_availability"])
    for incompatible_dataset in datasets_to_delete:
        del dataset_hyperparameters[incompatible_dataset]

    # Delete the 'target_availability' key
    compatible_datasets = tuple(dataset_hyperparameters.keys())
    for compatible_dataset in compatible_datasets:
        del dataset_hyperparameters[compatible_dataset]["target_availability"]

    # -----------------
    # Sample target scaler
    # -----------------
    scaler_hyperparameters = dict()
    for scaler, domain in config["Targets"][config["selected_target"]]["scaler"].items():
        if isinstance(domain, dict) and "dist" in domain:
            scaler_hyperparameters[scaler] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
        else:
            scaler_hyperparameters[scaler] = domain

    # -----------------
    # Sample training hyperparameters
    # -----------------
    train_hyperparameters = dict()
    for param, domain in config["Training"].items():
        if isinstance(domain, dict) and "dist" in domain:
            train_hyperparameters[param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
        else:
            train_hyperparameters[param] = domain
    train_hyperparameters["metrics"] = config["Targets"][config["selected_target"]]["metrics"]
    _activation_function = config["Targets"][config["selected_target"]]["prediction_activation_function"]
    train_hyperparameters["prediction_activation_function"] = _activation_function
    train_hyperparameters["main_metric"] = random.choice(config["Targets"][config["selected_target"]]["main_metric"])

    # -----------------
    # Sample loss
    # -----------------
    loss_config = config["Targets"][config["selected_target"]]["Loss"]
    loss_hyperparameters = dict()

    loss_hyperparameters["loss"] = sample_hyperparameter(loss_config["loss"]["dist"], **loss_config["loss"]["kwargs"])
    loss_hyperparameters["weighter"] = sample_hyperparameter(loss_config["weighter"]["dist"],
                                                             **loss_config["weighter"]["kwargs"])
    loss_hyperparameters["loss_kwargs"] = loss_config["loss_kwargs"]
    loss_hyperparameters["loss_kwargs"]["reduction"] = "mean" if loss_hyperparameters["weighter"] is None else "none"

    weighter_kwargs = dict()
    for param, domain in loss_config["weighter_kwargs"].items():
        if isinstance(domain, dict) and "dist" in domain:
            weighter_kwargs[param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
        else:
            weighter_kwargs[param] = domain

    # Add weighter kwargs to loss
    loss_hyperparameters["weighter_kwargs"] = weighter_kwargs

    # Add loss to training section in config file
    train_hyperparameters["Loss"] = loss_hyperparameters

    # -----------------
    # Sample method for handling a varied
    # number of channels
    # -----------------
    spatial_dimension_config = random.choice(config["Varied Numbers of Channels"])
    spatial_dimension_method = spatial_dimension_config["name"]
    if spatial_dimension_method == "RegionBasedPooling":
        varied_numbers_of_channels = {"name": spatial_dimension_method,
                                      "kwargs": sample_rbp_designs(spatial_dimension_config["kwargs"])}
        # config["Varied Numbers of Channels"][method]
    elif spatial_dimension_config["name"] == "Interpolation":
        # Todo: Not very elegant to have this piece of code here...
        varied_numbers_of_channels = {
            "name": spatial_dimension_method,
            "kwargs": {"method": random.choice(spatial_dimension_config["kwargs"]["method"]),
                       "main_channel_system": random.choice(spatial_dimension_config["kwargs"]["main_channel_system"])},
        }
    else:
        raise ValueError(f"Expected method for handling varied numbers of EEG channels to be either region based "
                         f"pooling or interpolation, but found {spatial_dimension_method}")

    # -----------------
    # Sample DL architecture and its hyperparameters
    # -----------------
    # Choose architecture
    mts_module_name = random.choice(tuple(config["MTS Module"].keys()))

    # Set hyperparameters
    mts_module_hyperparameters = {
        **config["MTS Module"][mts_module_name]["general"],
        **get_mts_module_type(mts_module_name).sample_hyperparameters(config["MTS Module"][mts_module_name]["sample"])
    }

    # Combine architecture name and hyperparameters in a dict
    dl_model = {"model": mts_module_name, "kwargs": mts_module_hyperparameters}

    # Maybe add CMMN and normalisation to DL architecture
    if spatial_dimension_method == "Interpolation":
        dl_model["normalise"] = random.choice(config["NormaliseInputs"])

        dl_model["CMMN"] = {"use_cmmn_layer": random.choice(config["CMMN"]["use_cmmn_layer"]),
                            "kwargs": {}}
        for param, domain in config["CMMN"]["kwargs"].items():
            if isinstance(domain, dict) and "dist" in domain:
                dl_model["CMMN"]["kwargs"][param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
            else:
                dl_model["CMMN"]["kwargs"][param] = domain

    # -----------------
    # Select data pre-processing version
    # -----------------
    # Select from the desired folder
    preprocessed_folder = random.choice(config["PreprocessedFolder"])

    # Make selection
    _available_versions = os.listdir(os.path.join(get_numpy_data_storage_path(), preprocessed_folder))
    available_versions = tuple(version for version in _available_versions if version[:5] == "data_")
    selected_version = random.choice(available_versions)

    # -----------------
    # Adjust the sampling frequency if required by the DL architecture
    # -----------------
    # Extract preprocessing config
    with open(os.path.join(get_numpy_data_storage_path(), preprocessed_folder, "config.yml"), "r") as file:
        pre_processed_config = yaml.safe_load(file)

    while True:
        # If the selected version does not exist, raise an error
        if selected_version not in available_versions:
            raise FailedModelInitialisationError(f"No attempted preprocessing version yielded a successful model "
                                                 f"initialisation ({mts_module_name}). Last attempt: "
                                                 f"{selected_version}")

        # Compute number of time steps
        filtering = selected_version.split("_")[3].split(sep="-")
        l_freq, h_freq = float(filtering[0]), float(filtering[1])
        s_freq = float(selected_version.split("_")[-1]) * h_freq

        num_time_steps = int(s_freq * pre_processed_config["general"]["epoch_duration"])

        if "num_time_steps" in dl_model["kwargs"]:
            dl_model["kwargs"]["num_time_steps"] = num_time_steps

        # If the number of time steps is too short for the architecture to handle, try doubling the sampling frequency.
        # If not possible, an error is raised upon the next iteration
        # (We just need to check if the inputs work, 'in_channels' argument is not really important)
        if get_mts_module_type(mts_module_name).successful_initialisation(in_channels=19, **dl_model["kwargs"]):
            break

        old_sampling_freq_multiple = selected_version.split("_")[-1]  # this will be a string
        new_sampling_freq_multiple = int(selected_version.split("_")[-1]) * 2
        print(f"Changing sampling rate from a multiple of {old_sampling_freq_multiple} to {new_sampling_freq_multiple}")
        selected_version = f"{selected_version[:-len(old_sampling_freq_multiple)]}{new_sampling_freq_multiple}"

    # Add details
    pre_processed_config["general"]["filtering"] = [l_freq, h_freq]
    pre_processed_config["general"]["autoreject"] = _str_to_bool(selected_version.split("_")[5])
    pre_processed_config["general"]["resample"] = float(selected_version.split("_")[-1]) * h_freq
    pre_processed_config["general"]["num_time_steps"] = int(s_freq * pre_processed_config["general"]["epoch_duration"])
    del pre_processed_config["frequency_bands"]
    del pre_processed_config["resample_fmax_multiples"]

    # Add preprocessing version to the datasets
    datasets = tuple(config["Datasets"])  # Maybe this is not needed, but it feels safe
    for dataset in datasets:
        config["Datasets"][dataset]["pre_processed_version"] = os.path.join(preprocessed_folder, selected_version)

    # -----------------
    # Domain discriminator
    # -----------------
    # Choose architecture
    discriminator_name = random.choice(tuple(config["DomainDiscriminator"]["discriminators"].keys()))

    discriminator: Optional[Dict[str, Any]]
    if discriminator_name != "NoDiscriminator" and config["cv_method"] != "inverted":
        # Architecture hyperparameters
        dd_structure = random.choice(
            tuple(config["DomainDiscriminator"]["discriminators"][discriminator_name].keys())
        )
        discriminator_architecture = {
            "name": discriminator_name,
            "kwargs": get_domain_discriminator_type(discriminator_name).sample_hyperparameters(
                dd_structure, config=config["DomainDiscriminator"]["discriminators"][discriminator_name][dd_structure],
                in_features=get_mts_module_type(mts_module_name).get_latent_features_dim(in_channels=19,
                                                                                         **dl_model["kwargs"])
            )}  # this will only work if the number of features is independent of number of input channels, otherwise an
        # error message will be raised during the experiment. Doctests have shown that number of input channels does
        # not affect latent feature dimension

        # Training hyperparameters
        discriminator = {"discriminator": discriminator_architecture, "training": dict()}
        for param, domain in config["DomainDiscriminator"]["training"].items():
            if isinstance(domain, dict) and "dist" in domain:
                discriminator["training"][param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
            else:
                discriminator["training"][param] = domain
    else:
        discriminator = None

    # Add method to train hyperparameters
    train_hyperparameters["method"] = "domain_discriminator_training" if discriminator is not None \
        else "downstream_training"

    # -----------------
    # Create final dictionaries
    # -----------------
    return ({"SubjectSplit": subjects_split_hyperparameters,
             "SubGroups": sub_groups_hyperparameters,
             "Datasets": dataset_hyperparameters,
             "Scalers": scaler_hyperparameters,
             "Training": train_hyperparameters,
             "Varied Numbers of Channels": varied_numbers_of_channels,
             "DL Architecture": dl_model,
             "DomainDiscriminator": discriminator,
             "run_baseline": config["run_baseline"],
             "cv_method": config["cv_method"],
             "LatentFeatureDistribution": config["LatentFeatureDistribution"]},
            pre_processed_config)


class FailedModelInitialisationError(Exception):
    ...
