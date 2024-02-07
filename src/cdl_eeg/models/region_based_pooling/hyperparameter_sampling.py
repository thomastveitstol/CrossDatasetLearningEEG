import copy
import random

from cdl_eeg.models.random_search.sampling_distributions import sample_hyperparameter


def sample_rbp_designs(config):
    """
    Function for generating multiple RBP designs

    Parameters
    ----------
    config : dict[str, typing.Any]
        Contains the domains of which to sample the RBP design choices from

    Returns
    -------
    dict[str, typing.Any]
    """
    # Sample the number of montage splits
    num_montage_splits = sample_hyperparameter(config["num_montage_splits"]["dist"],
                                               **config["num_montage_splits"]["kwargs"])

    # Sample the number of pooling modules. It cannot exceed the number of montage splits
    num_pooling_modules = 1 if random.choice(config["share_all_pooling_modules"]) \
        else random.choice([candidate_number for candidate_number in config["num_pooling_modules"]
                            if candidate_number <= num_montage_splits])

    # Randomly generate the number of montage splits per pooling module (they are partitioned, in a way)
    partitions = _generate_partition_sizes(n=num_montage_splits, k=num_pooling_modules)

    # Create all RBP designs
    designs = dict()
    for i, k in enumerate(partitions):
        # Generate a single RBP design
        designs[f"RBPDesign{i}"] = _sample_single_rbp_design(config=config["RBPDesign"], num_montage_splits=k)

    # Sample if the region representations should be normalised or not
    normalise = sample_hyperparameter(config["normalise_region_representations"]["dist"],
                                      **config["normalise_region_representations"]["kwargs"])

    return {"RBPDesigns": designs, "normalise_region_representations": normalise}


def _sample_single_rbp_design(config, num_montage_splits):
    """
    Function for sampling a single RBP design

    Parameters
    ----------
    config : dict
        Contains information on the domains to sample from
    num_montage_splits : int
        Number of montage splits for the current design

    Returns
    -------
    dict[str, typing.Any]
    """
    design = dict()

    # Number of designs
    design["num_designs"] = config["num_designs"]  # Should be 1

    # Pooling type
    design["pooling_type"] = config["pooling_type"]  # Should be multi_cs

    # ------------------
    # Pooling module (this works for current implementation)
    # ------------------
    pooling_method = random.choice(tuple(config["pooling_module"].keys()))
    design["pooling_methods"] = pooling_method
    design["pooling_methods_kwargs"] = dict()
    for pooling_kwarg, domain in config["pooling_module"][pooling_method].items():
        if isinstance(domain, list):
            design["pooling_methods_kwargs"][pooling_kwarg] = random.choice(domain)
        else:
            design["pooling_methods_kwargs"][pooling_kwarg] = domain

    # ------------------
    # Montage splits
    # ------------------
    # Generate multiple montage splits randomly
    montage_splits = tuple(random.choice(tuple(config["montage_split"].keys())) for _ in range(num_montage_splits))

    # Generate the hyperparameters of the montage splits
    montage_splits_kwargs = []
    for montage_split in montage_splits:
        montage_split_kwargs = dict()
        for split_kwarg, domain in config["montage_split"][montage_split].items():
            if isinstance(domain, list):
                montage_split_kwargs[split_kwarg] = random.choice(copy.deepcopy(domain))
            else:
                montage_split_kwargs[split_kwarg] = domain
        montage_splits_kwargs.append(montage_split_kwargs)

    # Add montage splits (both names and kwargs) to design
    design["split_methods"] = list(montage_splits)
    design["split_methods_kwargs"] = montage_splits_kwargs

    return design


def _generate_partition_sizes(*, n, k):
    """
    Function for randomly assigning cardinalities to subsets of a set of length n to partition. Any solution in positive
    integers to the of the equation x_1 + x_2 + ... + x_k = n is ok.

    Parameters
    ----------
    n : Number of montage splits
    k : Number of partitions

    Returns
    -------
    tuple[int, ...]

    Examples
    --------
    >>> random.seed(2)
    >>> _generate_partition_sizes(n=10, k=3)
    (5, 2, 3)

    The sum will always equal n

    >>> all(sum(_generate_partition_sizes(n=n_, k=k_)) == n_  # type: ignore[attr-defined]
    ...         for n_, k_ in zip((10, 20, 15, 64), (5, 10, 5, 33)))
    True
    """
    # Generate k 'cardinalities'
    cardinalities = [1 for _ in range(k)]

    # Iteratively increment the sizes
    for _ in range(n-k):
        # Increment a randomly selected cardinality
        cardinalities[random.randint(0, k-1)] += 1

    # Return as tuple
    return tuple(cardinalities)
