from typing import Type, Tuple

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.datasets.child_mind_dataset import ChildMind
from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.ous_dataset import OUS
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def get_dataset(dataset_name, **kwargs):
    """
    Function for getting the specified dataset

    Parameters
    ----------
    dataset_name : str
        Dataset name
    kwargs
        Key word arguments

    Returns
    -------
    EEGDatasetBase
    """
    # All available datasets must be included here
    available_datasets: Tuple[Type[EEGDatasetBase], ...] = (Miltiadous, HatlestadHall, YulinWang, OUS, CAUEEG, MPILemon,
                                                            ChildMind)

    # Loop through and select the correct one
    for dataset in available_datasets:
        if dataset_name in (dataset.__name__, dataset().name):
            return dataset(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The dataset '{dataset_name}' was not recognised. Please select among the following: "
                     f"{tuple(dataset.__name__ for dataset in available_datasets)}")


def get_channel_system(dataset_name, **kwargs):
    """
    Function for getting the specified channel system

    Parameters
    ----------
    dataset_name : str
        Dataset name
    kwargs
        Keyword arguments

    Returns
    -------
    cdl_eeg.data.datasets.dataset_base.ChannelSystem
    """
    return get_dataset(dataset_name, **kwargs).channel_system
