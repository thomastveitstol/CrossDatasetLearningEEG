import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase
from cdl_eeg.data.datasets.getter import get_dataset


# -----------------
# Convenient dataclasses
# -----------------
@dataclasses.dataclass(frozen=True)
class LoadDetails:
    subject_ids: Tuple[str, ...]
    time_series_start: Optional[int] = None
    num_time_steps: Optional[int] = None
    channels: Optional[Tuple[str, ...]] = None


# -----------------
# Classes
# -----------------
class CombinedDatasets:
    """
    Class for storing multiple datasets

    TODO: Make methods for introducing more data, and for removing from memory
    """

    __slots__ = "_subject_ids", "_data", "_targets", "_datasets"

    def __init__(self, datasets, load_details=None, target=None):
        """
        Initialise

        (unittest in test folder)

        Parameters
        ----------
        datasets : tuple[EEGDatasetBase, ...]
        load_details : tuple[LoadDetails, ...], optional
        target: str, optional
            Targets to load. If None, no targets are loaded
        """
        # If no loading details are provided, use default
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
                                                              num_time_steps=details.num_time_steps,
                                                              channels=details.channels)
                      for dataset, details in zip(datasets, load_details)}

        self._targets = None if target is None \
            else {dataset.name: dataset.load_targets(subject_ids=details.subject_ids, target=target)
                  for dataset, details in zip(datasets, load_details)}

        # Convenient for e.g. extracting channel systems
        self._datasets: Tuple[EEGDatasetBase, ...] = datasets

    @classmethod
    def from_config(cls, config, target=None):
        """
        Method for initialising directly from a config file

        Parameters
        ----------
        config : dict[str, typing.Any]
        target : str, optional

        Returns
        -------
        """
        # Initialise lists and dictionaries
        load_details = []
        datasets = []
        subjects = dict()
        channel_name_to_index = dict()

        # Loop through all datasets and loading details to be used
        for dataset_name, dataset_details in config.items():
            # Get dataset
            dataset = get_dataset(dataset_name)
            datasets.append(dataset)
            dataset_subjects = dataset.get_subject_ids()[:dataset_details["num_subjects"]]
            subjects[dataset_name] = dataset_subjects
            channel_name_to_index[dataset_name] = dataset.channel_name_to_index()

            # Construct loading details
            load_details.append(
                LoadDetails(subject_ids=dataset_subjects, time_series_start=dataset_details["time_series_start"],
                            num_time_steps=dataset_details["num_time_steps"])
            )

        # Load all data and return object
        return cls(datasets=tuple(datasets), load_details=tuple(load_details), target=target)

    def get_data(self, subjects):
        """
        Method for getting data

        (unittest in test folder)

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
        return {dataset_name: numpy.concatenate(numpy.expand_dims(data_matrix, axis=0), axis=0)
                for dataset_name, data_matrix in data.items()}

    def get_targets(self, subjects):
        """
        Method for getting targets

        TODO: make tests

        Parameters
        ----------
        subjects : tuple[cdl_eeg.data.data_split.Subject, ...]
            Subjects to extract

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        # Input check
        if self._targets is None:
            raise ValueError("Tried to extract targets, but no targets are available")

        # Loop through all subjects  todo: fix type hinting
        data: Dict[str, List[numpy.ndarray]] = dict()  # type: ignore[type-arg]
        for subject in subjects:
            dataset_name = subject.dataset_name

            # Get the target
            idx = self._subject_ids[dataset_name][subject.subject_id]
            subject_data = self._targets[dataset_name][idx]

            # Add the subject data
            if dataset_name in data:
                data[dataset_name].append(subject_data)
            else:
                data[dataset_name] = [subject_data]

        # Convert to numpy arrays and return (here, we assume that the data matrices can be concatenated)
        return {dataset_name: numpy.concatenate(numpy.expand_dims(data_matrix, axis=0), axis=0)
                for dataset_name, data_matrix in data.items()}

    @staticmethod
    def get_subjects_dict(subjects):
        """
        Method for the subjects as a dictionary

        Parameters
        ----------
        subjects : tuple[cdl_eeg.data.data_split.Subject, ...]
            Subjects to extract

        Returns
        -------
        dict[str, tuple[str, ...]]

        Examples
        --------
        >>> from cdl_eeg.data.data_split import Subject
        >>> my_drivers = (Subject(dataset_name="Merc", subject_id="LH"), Subject(dataset_name="RB", subject_id="SP"),
        ...               Subject(dataset_name="AM", subject_id="FA"), Subject(dataset_name="RB", subject_id="MV"),
        ...               Subject(dataset_name="Merc", subject_id="GR"))
        >>> CombinedDatasets.get_subjects_dict(my_drivers)
        {'Merc': ('LH', 'GR'), 'RB': ('SP', 'MV'), 'AM': ('FA',)}
        """
        subjects_dict = dict()
        for subject in subjects:
            # Add the subject data
            if subject.dataset_name in subjects_dict:
                subjects_dict[subject.dataset_name].append(subject.subject_id)
            else:
                subjects_dict[subject.dataset_name] = [subject.subject_id]
        return {dataset_name: tuple(subject_ids) for dataset_name, subject_ids in subjects_dict.items()}

    # ----------------
    # Properties
    # ----------------
    @property
    def dataset_subjects(self) -> Dict[str, Tuple[str, ...]]:
        """Get a dictionary containing the subjects available (values) in the datasets (keys)"""
        return {name: tuple(subjects.keys()) for name, subjects in self._subject_ids.items()}

    @property
    def datasets(self):
        return self._datasets

    @property
    def channel_name_to_index(self):
        return {dataset.name: dataset.channel_name_to_index() for dataset in self._datasets}

# todo: check out asyncio for loading. See mCoding at https://www.youtube.com/watch?v=ftmdDlwMwwQ and
#  https://www.youtube.com/watch?v=ueTXYhtlnjA
