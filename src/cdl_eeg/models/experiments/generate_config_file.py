import random

from cdl_eeg.models.hyperparameter_sampling import sample_rbp_designs


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
    for hyperparameter_name, hyperparameter_domain in config["MTS Module"][mts_module_name].items():
        if isinstance(hyperparameter_domain, list):
            mts_module_hyperparameters[hyperparameter_name] = random.choice(hyperparameter_domain)
        else:
            mts_module_hyperparameters[hyperparameter_name] = hyperparameter_domain

    # Combine architecture name and hyperparameters in a dict
    dl_model = {"model": mts_module_name, "kwargs": mts_module_hyperparameters}

    # -----------------
    # Domain discriminator
    # -----------------
    # Choose architecture
    discriminator_name = random.choice(tuple(config["DomainDiscriminator"]["discriminators"].keys()))

    if discriminator_name != "NoDiscriminator":
        discriminator_architecture = {"name": discriminator_name, "kwargs": dict()}
        for hyperparameter_name, hyperparameter_domain \
                in config["DomainDiscriminator"]["discriminators"][discriminator_name].items():
            if isinstance(hyperparameter_domain, list):
                discriminator_architecture["kwargs"][hyperparameter_name] = random.choice(hyperparameter_domain)
            else:
                discriminator_architecture["kwargs"][hyperparameter_name] = hyperparameter_domain

        # Training hyperparams
        discriminator = {"discriminator": discriminator_architecture, "training": dict()}
        for hyperparameter_name, hyperparameter_domain in config["DomainDiscriminator"]["training"].items():
            if isinstance(hyperparameter_domain, list):
                discriminator["training"][hyperparameter_name] = random.choice(hyperparameter_domain)
            else:
                discriminator["training"][hyperparameter_name] = hyperparameter_domain
    else:
        discriminator = None

    # -----------------
    # Save as .yml file
    # -----------------
    return {"Training": train_hyperparameters, "Varied Numbers of Channels": varied_numbers_of_channels,
            "DL Architecture": dl_model, "DomainDiscriminator": discriminator}
