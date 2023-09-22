import dataclasses
import random
from typing import Tuple, List

import numpy


@dataclasses.dataclass(frozen=True)
class Subject:
    subject_id: str
    dataset_name: str


class KFoldDataSplit:
    """
    Class for splitting the data into k folds. The different datasets are neglected

    Examples
    --------
    >>> f1_drivers = {"Mercedes": ("Hamilton", "Russel", "Wolff"), "Red Bull": ("Verstappen", "Checo"),
    ...               "Ferrari": ("Leclerc", "Smooth Sainz"), "McLaren": ("Norris", "Piastri"),
    ...               "Aston Martin": ("Alonso", "Stroll"), "Alpine": ("Magnussen", "Hülkenberg")}
    >>> de_vries_points = 0
    >>> my_folds = KFoldDataSplit(num_folds=3, dataset_subjects=f1_drivers, seed=de_vries_points).folds
    >>> my_folds  # doctest: +NORMALIZE_WHITESPACE
    ((Subject(subject_id='Russel', dataset_name='Mercedes'), Subject(subject_id='Stroll', dataset_name='Aston Martin'),
      Subject(subject_id='Alonso', dataset_name='Aston Martin'), Subject(subject_id='Leclerc', dataset_name='Ferrari'),
      Subject(subject_id='Magnussen', dataset_name='Alpine')),
     (Subject(subject_id='Wolff', dataset_name='Mercedes'), Subject(subject_id='Verstappen', dataset_name='Red Bull'),
      Subject(subject_id='Norris', dataset_name='McLaren'), Subject(subject_id='Piastri', dataset_name='McLaren')),
     (Subject(subject_id='Checo', dataset_name='Red Bull'), Subject(subject_id='Hamilton', dataset_name='Mercedes'),
      Subject(subject_id='Hülkenberg', dataset_name='Alpine'),
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
