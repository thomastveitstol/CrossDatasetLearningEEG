import dataclasses
from typing import Optional, Tuple


@dataclasses.dataclass(frozen=True)
class LoadDetails:
    subject_ids: Tuple[str, ...]
    time_series_start: Optional[int] = None
    num_time_steps: Optional[int] = None


class CombinedDatasets:

    def __init__(self, datasets, load_details=None):
        """
        Initialise

        Parameters
        ----------
        datasets : tuple[cdl_eeg.data.datasets.dataset_base.EEGDatasetBase]
        load_details : tuple[LoadDetails, ...], optional
        """
        load_details = tuple(LoadDetails(dataset.get_subject_ids()) for dataset in datasets) \
            if load_details is None else load_details

        # --------------
        # Input check
        # --------------
        if len(datasets) != len(load_details):
            raise ValueError(f"Expected number of datasets to be the same as the number of loading details, but found "
                             f"{len(datasets)} and {len(load_details)}")
        # --------------
        # Store attributes
        # --------------
        # Store subject IDs
        self._subject_ids = {dataset.name: details.subject_ids for dataset, details in zip(datasets, load_details)}

        # Load and store data
        self._data = {dataset.name: dataset.load_numpy_arrays(subject_ids=details.subject_ids,
                                                              time_series_start=details.time_series_start,
                                                              num_time_steps=details.num_time_steps)
                      for dataset, details in zip(datasets, load_details)}

    def get_data(self, subjects):
        """
        Method for getting data

        Parameters
        ----------
        subjects : tuple[cdl_eeg.data.data_split.Subject, ...]

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        raise NotImplementedError
