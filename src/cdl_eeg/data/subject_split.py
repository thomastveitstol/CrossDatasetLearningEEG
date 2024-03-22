import abc
import dataclasses
import itertools
import random
from typing import List, Tuple

import numpy


# -----------------
# Convenient dataclasses
# -----------------
@dataclasses.dataclass(frozen=True)
class Subject:
    """
    Class for defining a subject. Convenient particularly when different datasets use the same subject IDs

    Examples
    --------
    >>> Subject("Person", "Dataset")
    Subject(subject_id='Person', dataset_name='Dataset')

    Can be used as keys in a dict

    >>> my_subject = {Subject("P1", "D1"): "this_is_a_value"}
    >>> my_subject[Subject("P1", "D1")]
    'this_is_a_value'

    Attributes can also be obtained as if the class was a dict

    >>> Subject("P1", "D1")["dataset_name"]
    'D1'
    """
    subject_id: str
    dataset_name: str

    def __getitem__(self, item):
        return getattr(self, item)


# -----------------
# Base classes
# -----------------
class DataSplitBase(abc.ABC):

    __slots__ = ()

    @property
    @abc.abstractmethod
    def folds(self):
        """
        Get the folds

        Returns
        -------
        tuple[tuple[Subject, ...], ...]
        """


# -----------------
# Classes
# -----------------
class KFoldDataSplit(DataSplitBase):
    """
    Class for splitting the data into k folds. The different datasets are neglected

    Examples
    --------
    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    >>> de_vries_points = 0
    >>> my_folds = KFoldDataSplit(num_folds=3, dataset_subjects=f1_drivers, seed=de_vries_points).folds
    >>> my_folds  # doctest: +NORMALIZE_WHITESPACE
    ((Subject(subject_id='Russel', dataset_name='Mercedes'), Subject(subject_id='Stroll', dataset_name='Aston Martin'),
      Subject(subject_id='Alonso', dataset_name='Aston Martin'), Subject(subject_id='Leclerc', dataset_name='Ferrari'),
      Subject(subject_id='Magnussen', dataset_name='Haas')),
     (Subject(subject_id='Wolff', dataset_name='Mercedes'), Subject(subject_id='Verstappen', dataset_name='Red Bull'),
      Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren')),
     (Subject(subject_id='Checo', dataset_name='Red Bull'), Subject(subject_id='Hamilton', dataset_name='Mercedes'),
      Subject(subject_id='Hülkenberg', dataset_name='Haas'),
      Subject(subject_id='Smooth Sainz', dataset_name='Ferrari')))
    """

    __slots__ = "_folds",

    def __init__(self, *, num_folds, dataset_subjects, seed=None):
        """
        Initialise

        Parameters
        ----------
        num_folds : int
            Number of folds
        dataset_subjects : dict[str, tuple[str, ...]]
            Subject IDs. The keys are dataset names, the values are the subject IDs of the corresponding dataset
        seed : int, optional
            Seed for making the data split reproducible. If None, no seed is set

        """
        # Pool all subjects together
        subjects = []
        for dataset_name, subject_ids in dataset_subjects.items():
            for sub_id in subject_ids:
                subjects.append(Subject(subject_id=sub_id, dataset_name=dataset_name))

        # Maybe make data split reproducible
        if seed is not None:
            random.seed(seed)

        # Shuffle
        random.shuffle(subjects)

        # Perform split
        split = numpy.array_split(subjects, num_folds)  # type: ignore[arg-type, var-annotated]

        # Set attribute (and some type fix, type hinting and mypy stuff)
        folds: List[Tuple[Subject, ...]] = []
        for fold in split:
            folds.append(tuple(fold))
        self._folds = tuple(folds)

    # ---------------
    # Properties
    # ---------------
    @property
    def folds(self):
        return self._folds


class SplitOnDataset(DataSplitBase):
    """
    Class for splitting the data based on the provided datasets only

    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Haas": ("Magnussen", "Hülkenberg")}
    >>> de_vries_points = 0
    >>> my_folds = SplitOnDataset(dataset_subjects=f1_drivers, seed=de_vries_points).folds
    >>> my_folds  # doctest: +NORMALIZE_WHITESPACE
    ((Subject(subject_id='Magnussen', dataset_name='Haas'), Subject(subject_id='Hülkenberg', dataset_name='Haas')),
     (Subject(subject_id='Hamilton', dataset_name='Mercedes'), Subject(subject_id='Wolff', dataset_name='Mercedes'),
      Subject(subject_id='Russel', dataset_name='Mercedes')),
     (Subject(subject_id='Alonso', dataset_name='Aston Martin'), Subject(subject_id='Stroll',
                                                                         dataset_name='Aston Martin')),
     (Subject(subject_id='Checo', dataset_name='Red Bull'), Subject(subject_id='Verstappen', dataset_name='Red Bull')),
     (Subject(subject_id='Leclerc', dataset_name='Ferrari'), Subject(subject_id='Smooth Sainz',
                                                                     dataset_name='Ferrari')),
     (Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren')))
    """

    def __init__(self, dataset_subjects, *, seed=None):
        """
        Initialise

        Parameters
        ----------
        dataset_subjects : dict[str, tuple[str, ...]]
            Subject IDs. The keys are dataset names, the values are the subject IDs of the corresponding dataset
        seed : int, optional
            Seed for making the data split reproducible. If None, no seed is set
        """
        # Maybe make data split reproducible
        if seed is not None:
            random.seed(seed)

        # Loop though the datasets
        folds = []
        for dataset_name, subject_ids in dataset_subjects.items():
            # Fix type
            sub_ids = [Subject(dataset_name=dataset_name, subject_id=subject_id) for subject_id in subject_ids]

            # Shuffle
            random.shuffle(sub_ids)

            # Add it as a tuple to the folds
            folds.append(tuple(sub_ids))

        # Shuffle the folds (likely not necessary, but why not)
        random.shuffle(folds)

        # Set attribute
        self._folds = tuple(folds)

    # ---------------
    # Properties
    # ---------------
    @property
    def folds(self):
        return self._folds


class TrainValBase(DataSplitBase, abc.ABC):
    """
    Base class when slitting into training and validation
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def folds(self):
        """
        Get the folds. The first 'fold' is meant for training, the last for validation. Not the nicest way to do things,
        so todo: this could benefit from a refactoring

        Returns
        -------
        tuple[tuple[Subject, ...], tuple[Subject, ...]]
        """


class DatasetBalancedTrainValSplit(TrainValBase):
    """
    Ensure that the training and validation is split per dataset

    Examples
    --------
    >>> my_subjects = {"d1": tuple(f"s{i}" for i in range(10)), # type: ignore[attr-defined]
    ...                "d2": tuple(f"s{i}" for i in range(20)),  # type: ignore[attr-defined]
    ...                "d3": tuple(f"s{i}" for i in range(15))}  # type: ignore[attr-defined]
    >>> my_folds = DatasetBalancedTrainValSplit(my_subjects, val_split=0.2, seed=2).folds

    Check dataset sizes in train and validation set

    >>> {dataset: len(tuple(sub for sub in my_folds[0] if sub.dataset_name == dataset))   # type: ignore[attr-defined]
    ...  for dataset in my_subjects}
    {'d1': 8, 'd2': 16, 'd3': 12}
    >>> {dataset: len(tuple(sub for sub in my_folds[1] if sub.dataset_name == dataset))   # type: ignore[attr-defined]
    ...  for dataset in my_subjects}
    {'d1': 2, 'd2': 4, 'd3': 3}

    Train and validation are not overlapping

    >>> any(sub in my_folds[0] for sub in my_folds[1])  # type: ignore[attr-defined]
    False
    """

    __slots__ = "_train_subjects", "_val_subjects"

    def __init__(self, dataset_subjects, *, val_split, seed=None):
        # Maybe make data split reproducible
        if seed is not None:
            random.seed(seed)

        # Loop through all datasets
        train_subjects: List[Subject] = []
        val_subjects: List[Subject] = []
        for dataset, subjects in dataset_subjects.items():
            dataset_train_subjects, dataset_val_subjects = _split_randomly(subjects=subjects, split_percent=val_split)

            train_subjects.extend([Subject(subject_id=sub_id, dataset_name=dataset)
                                   for sub_id in dataset_train_subjects])
            val_subjects.extend([Subject(subject_id=sub_id, dataset_name=dataset) for sub_id in dataset_val_subjects])

        self._train_subjects = tuple(train_subjects)
        self._val_subjects = tuple(val_subjects)

    @property
    def folds(self):
        return self._train_subjects, self._val_subjects


# -----------------
# Functions
# -----------------
def get_data_split(split, **kwargs):
    """
    Function for getting the specified data split

    Parameters
    ----------
    split : str
    kwargs

    Returns
    -------
    DataSplitBase
    """
    # All available data splits must be included here
    available_splits = (KFoldDataSplit, SplitOnDataset)

    # Loop through and select the correct one
    for split_class in available_splits:
        if split == split_class.__name__:
            return split_class(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The data split '{split}' was not recognised. Please select among the following: "
                     f"{tuple(split_class.__name__ for split_class in available_splits)}")


def leave_1_fold_out(i, folds):
    """
    Method for selecting all subject except for one fold (the i-th fold)

    Parameters
    ----------
    i : int
        Fold to not include
    folds : tuple[tuple[Subject, ...], ...]

    Returns
    -------
    tuple[Subject, ...]

    Examples
    --------
    >>> my_folds = ((Subject("TW", "Merc"), Subject("MV", "RB"), Subject("LN", "McL")),
    ...             (Subject("YT", "AT"), Subject("CS", "F")), (Subject("CL", "F"), Subject("VB", "AR")),
    ...             (Subject("FA", "AM"), Subject("LS", "AM"), Subject("DH", "RB")))
    >>> leave_1_fold_out(2, my_folds)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='TW', dataset_name='Merc'), Subject(subject_id='MV', dataset_name='RB'),
     Subject(subject_id='LN', dataset_name='McL'), Subject(subject_id='YT', dataset_name='AT'),
     Subject(subject_id='CS', dataset_name='F'), Subject(subject_id='FA', dataset_name='AM'),
     Subject(subject_id='LS', dataset_name='AM'), Subject(subject_id='DH', dataset_name='RB'))
    >>> leave_1_fold_out(-1, my_folds)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='TW', dataset_name='Merc'), Subject(subject_id='MV', dataset_name='RB'),
     Subject(subject_id='LN', dataset_name='McL'), Subject(subject_id='YT', dataset_name='AT'),
     Subject(subject_id='CS', dataset_name='F'), Subject(subject_id='CL', dataset_name='F'),
     Subject(subject_id='VB', dataset_name='AR'))

    """
    # Handle negative index
    i = len(folds) + i if i < 0 else i

    # Return as unpacked tuple
    return tuple(itertools.chain(*tuple(fold for j, fold in enumerate(folds) if j != i)))


def _split_randomly(subjects, split_percent):
    # Input checks
    assert all(isinstance(subject, (Subject, str)) for subject in subjects)
    assert isinstance(split_percent, float)
    assert 0 < split_percent < 1

    # Make a list and a copy
    subjects = list(subjects)

    # Shuffle randomly
    random.shuffle(subjects)

    # Split by the percentage
    num_subjects = len(subjects)
    split_idx = int(num_subjects * (1 - split_percent))

    return subjects[:split_idx], subjects[split_idx:]
