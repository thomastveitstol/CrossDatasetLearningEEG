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
