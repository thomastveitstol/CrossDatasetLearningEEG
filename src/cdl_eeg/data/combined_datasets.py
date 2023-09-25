import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy


# -----------------
# Convenient dataclasses
# -----------------
@dataclasses.dataclass(frozen=True)
class LoadDetails:
    subject_ids: Tuple[str, ...]
    time_series_start: Optional[int] = None
    num_time_steps: Optional[int] = None


# -----------------
# Classes
# -----------------
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
        # Store subject IDs. Organised as {dataset_name: {subject_name: row-number in data matrix}}
        subject_ids: Dict[str, Dict[str, int]] = dict()
        for dataset, details in zip(datasets, load_details):
            subject_ids[dataset.name] = {sub_id: i for i, sub_id in enumerate(details.subject_ids)}

        self._subject_ids = subject_ids

        # Load and store data  todo: can this be made faster be asyncio?
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
            Subjects to extract

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        # Loop through all subjects  todo: fix type hinting
        data: Dict[str, List[numpy.ndarray]] = dict()  # type: ignore[type-arg]
        for subject in subjects:
            dataset_name = subject.dataset_name

            # Get the data
            idx = self._subject_ids[dataset_name][subject.subject_id]
            subject_data = self._data[dataset_name][idx]

            # Add the subject data
            if dataset_name in data:
                data[dataset_name].append(subject_data)
            else:
                data[dataset_name] = [subject_data]

        # Convert to numpy arrays and return (here, we assume that the data matrices can be concatenated)
        return {dataset_name: numpy.concatenate(data_matrix, axis=0) for dataset_name, data_matrix in data.items()}
