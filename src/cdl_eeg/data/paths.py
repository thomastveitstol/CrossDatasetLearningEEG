"""
Functions for getting the path to different files related to the data and results
"""
import json
import os


def get_results_dir():
    """
    Get the path to where the results of the experiments are to be stored

    Returns
    -------
    str

    Examples
    --------
    >>> get_results_dir()  # doctest: +ELLIPSIS
    '.../CrossDatasetLearningEEG/src/cdl_eeg/data/results'
    """
    return os.path.join(os.path.dirname(__file__), "results")


def get_raw_data_storage_path():
    """
    Get the path to where the newly downloaded data is (supposed to be) stored. It should be specified in the
    'config_paths.json' file

    Returns
    -------
    str
        The path to where the data is stored, or to be stored (e.g. in the scripts for downloading)
    """
    # Load the config file for paths
    config_path = os.path.join(os.path.dirname(__file__), "config_paths.json")
    with open(config_path) as f:
        config = json.load(f)

    return config["MNEPath"]


def get_numpy_data_storage_path():
    """
    Get the path to where the downloaded data is (supposed to be) stored as numpy arrays.

    Note that this will only work for me (Thomas). I could not store the datasets inside this Python-project, as they
    require too much memory
    Returns
    -------
    str
        The path to where the data is stored as numpy arrays, or to be stored (e.g. in the scripts for saving as numpy
        arrays)
    """
    return os.path.join(get_raw_data_storage_path(), "numpy_arrays")
