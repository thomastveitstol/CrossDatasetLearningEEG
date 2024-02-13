import random

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
    # Add dataset details
    # -----------------
    dataset_hyperparameters = config["Datasets"]

    # -----------------
    # Sample input and target scalers
    # -----------------
    scaler_hyperparameters = dict()
    for scaler, domain in config["Scalers"].items():
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

    # -----------------
    # Sample loss
    # -----------------
    loss_config = config["Loss"]
    loss_hyperparameters = dict()

    loss_hyperparameters["loss_name"] = sample_hyperparameter(loss_config["loss_name"]["dist"],
                                                              **loss_config["loss_name"]["kwargs"])
    loss_hyperparameters["weighter"] = sample_hyperparameter(loss_config["weighter"]["dist"],
                                                             **loss_config["weighter"]["kwargs"])
    loss_hyperparameters["loss_kwargs"] = "mean" if loss_hyperparameters["weighter"] is None else "none"

    weighter_kwargs = dict()  # todo: hard-coded :(
    for param, domain in weighter_kwargs.items():
        if isinstance(domain, dict) and "dist" in domain:
            weighter_kwargs[param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
        else:
            weighter_kwargs[param] = domain

    # Add weighter kwargs to loss
    loss_hyperparameters["Loss"]["weighter_kwargs"] = weighter_kwargs

    # Add loss to training section in config file
    train_hyperparameters["Loss"] = loss_hyperparameters

    # -----------------
    # Sample method for handling a varied
    # number of channels
    # -----------------
    varied_numbers_of_channels = dict()
    method = random.choice(tuple(config["Varied Numbers of Channels"].keys()))
    if method == "RegionBasedPooling":
        varied_numbers_of_channels[method] = sample_rbp_designs(
            config["Varied Numbers of Channels"][method]
        )
        # config["Varied Numbers of Channels"][method]
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
    for param, domain in config["MTS Module"][mts_module_name].items():
        if isinstance(domain, dict) and "dist" in domain:
            mts_module_hyperparameters[param] = sample_hyperparameter(domain["dist"], **domain["kwargs"])
        else:
            mts_module_hyperparameters[param] = domain

    # Combine architecture name and hyperparameters in a dict
    dl_model = {"model": mts_module_name, "kwargs": mts_module_hyperparameters}

    # -----------------
    # Domain discriminator
    # -----------------
    # Choose architecture
    discriminator_name = random.choice(tuple(config["DomainDiscriminator"]["discriminators"].keys()))

    if discriminator_name != "NoDiscriminator":
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

    # -----------------
    # Create final dictionary
    # -----------------
    return {"SubjectSplit": subjects_split_hyperparameters,
            "Datasets": dataset_hyperparameters,
            "Scalers": scaler_hyperparameters,
            "Training": train_hyperparameters,
            "Varied Numbers of Channels": varied_numbers_of_channels,
            "DL Architecture": dl_model,
            "DomainDiscriminator": discriminator}
