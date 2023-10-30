from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
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
    cdl_eeg.data.datasets.dataset_base.EEGDatasetBase
    """
    # All available datasets must be included here
    available_datasets = (Miltiadous, Rockhill, HatlestadHall, YulinWang)

    # Loop through and select the correct one
    for dataset in available_datasets:
        if dataset_name in (dataset.__name__, dataset().name):
            return dataset(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The dataset '{dataset_name}' was not recognised. Please select among the following: "
                     f"{tuple(dataset.__name__ for dataset in available_datasets)}")
