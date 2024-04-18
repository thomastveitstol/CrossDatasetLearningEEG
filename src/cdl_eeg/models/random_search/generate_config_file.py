import random
from typing import Any, Dict, Optional

from cdl_eeg.models.mts_modules.getter import get_mts_module_type
from cdl_eeg.models.region_based_pooling.hyperparameter_sampling import sample_rbp_designs
from cdl_eeg.models.random_search.sampling_distributions import sample_hyperparameter


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

    weighter_kwargs = dict()  # todo: hard-coded :(
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
    # Domain discriminator
    # -----------------
    # Choose architecture
    discriminator_name = random.choice(tuple(config["DomainDiscriminator"]["discriminators"].keys()))

    discriminator: Optional[Dict[str, Any]]
    if discriminator_name != "NoDiscriminator" and config["cv_method"] != "inverted":
        # Architecture hyperparameters
        discriminator_architecture = {"name": discriminator_name, "kwargs": dict()}
        for param, domain in config["DomainDiscriminator"]["discriminators"][discriminator_name].items():
            if isinstance(domain, dict) and "dist" in domain:
                discriminator_architecture["kwargs"][param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
            else:
                discriminator_architecture["kwargs"][param] = domain

        # Training hyperparameters
        discriminator = {"discriminator": discriminator_architecture, "training": dict()}
        for param, domain in config["DomainDiscriminator"]["training"].items():
            if isinstance(domain, dict) and "dist" in domain:
                discriminator["training"][param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
            else:
                discriminator["training"][param] = domain
    else:
        discriminator = None

    # Add method to train hyperparameters  todo: unelegant
    train_hyperparameters["method"] = "domain_discriminator_training" if discriminator is not None \
        else "downstream_training"

    # -----------------
    # Create final dictionary
    # -----------------
    return {"SubjectSplit": subjects_split_hyperparameters,
            "SubGroups": sub_groups_hyperparameters,
            "Datasets": dataset_hyperparameters,
            "Scalers": scaler_hyperparameters,
            "Training": train_hyperparameters,
            "Varied Numbers of Channels": varied_numbers_of_channels,
            "DL Architecture": dl_model,
            "DomainDiscriminator": discriminator,
            "run_baseline": config["run_baseline"],
            "cv_method": config["cv_method"],
            "LatentFeatureDistribution": config["LatentFeatureDistribution"]}
