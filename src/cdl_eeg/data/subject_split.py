import dataclasses
from typing import Any, Tuple, Dict

from cdl_eeg.data.data_split import Subject


# todo: I think this .py file should be removed
# -----------------
# Convenient dataclasses
# -----------------
@dataclasses.dataclass(frozen=True)
class Criterion:
    """Criterion for a group within a split. (This is important mainly because it may be used as a key, as opposed to
    mutable objects)"""
    crit: Any


# -----------------
# Functions
# -----------------
def filter_subjects(subjects, inclusion_criteria):
    """
    Function for filtering out subjects not satisfying the provided inclusion criteria

    Parameters
    ----------
    subjects: tuple[cdl_eeg.data.data_split.Subject, ...]
    inclusion_criteria : dict[str, tuple]

    Returns
    -------
    tuple[cdl_eeg.data.data_split.Subject, ...]

    Examples
    --------
    >>> my_subjects = (
    ...     Subject("P1", "D1"),
    ...     Subject("P2", "D1"),
    ...     Subject("P3", "D1"),
    ...     Subject("P1", "D2"),
    ...     Subject("P2", "D2"),
    ...     Subject("P1", "D3"),
    ...     Subject("P2", "D3"),
    ...     Subject("P3", "D3")
    >>> my_criteria = {"dataset_name": ("D1", "D2")}
    >>> filter_subjects(my_subjects, inclusion_criteria=my_criteria)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='P3', dataset_name='D1', details={'sex': 'female', 'age': 'young'}),
     Subject(subject_id='P3', dataset_name='D3', details={'sex': 'female', 'age': 'young'}))

    Multiple inclusion criteria is supported

    >>> my_criteria = {"sex": ("female", "male",), "age": ("old",)}
    >>> filter_subjects(my_subjects, inclusion_criteria=my_criteria)  # doctest: +NORMALIZE_WHITESPACE
    (Subject(subject_id='P1', dataset_name='D1', details={'sex': 'male', 'age': 'old'}),
     Subject(subject_id='P2', dataset_name='D1', details={'sex': 'male', 'age': 'old'}),
     Subject(subject_id='P1', dataset_name='D2', details={'sex': 'female', 'age': 'old'}),
     Subject(subject_id='P1', dataset_name='D3', details={'sex': 'male', 'age': 'old'}),
     Subject(subject_id='P2', dataset_name='D3', details={'sex': 'female', 'age': 'old'}))

    To split on dataset, it must be included in the details

    >>> my_criteria = {"dataset_name": ("D1", "D3")}
    >>> filter_subjects(my_subjects, inclusion_criteria=my_criteria)
    ()
    """
    # Initialise list of included subjects
    included_subjects = []

    # Loop through all subjects
    for subject in subjects:
        # Loop through all inclusion criteria. They must all be fulfilled to be included
        for filter_, criterion in inclusion_criteria.items():
            # Check current inclusion criteria and break if not satisfied
            # todo: sub optimal for e.g. age range
            if filter_ not in subject.details or subject.details[filter_] not in criterion:
                break
        else:
            # Never seen for-else? This will run if and only if the for-loop was finished without breaking or exception.
            # You may check it out in this video by mCoding (I know he doesn't recommend it, but I kinda liked it in
            # this case): https://www.youtube.com/watch?v=6Im38sF-sjo&t=283s
            included_subjects.append(subject)

    return tuple(included_subjects)


def make_subject_splits(subjects, splits):
    """
    Function for splitting subjects into different groups

    Parameters
    ----------
    subjects : subjects: tuple[cdl_eeg.data.data_split.Subject, ...]
    splits : dict[str, tuple]

    Returns
    -------
    dict[str, dict[Condition, tuple[cdl_eeg.data.data_split.Subject, ...]]]
    # todo: consider extending Condition dataclass to include the split and the included subjects

    Examples
    --------
    >>> my_subjects = (
    ...     Subject("P1", "D1", details={"sex": "male", "cognition": "hc"}),
    ...     Subject("P2", "D1", details={"sex": "male", "cognition": "hc"}),
    ...     Subject("P3", "D1", details={"sex": "female", "cognition": "mci"}),
    ...     Subject("P1", "D2", details={"sex": "female", "cognition": "ad"}),
    ...     Subject("P2", "D2", details={"sex": "male", "cognition": "ad"}),
    ...     Subject("P1", "D3", details={"sex": "male", "cognition": "mci"}),
    ...     Subject("P2", "D3", details={"sex": "female", "cognition": "hc"}),
    ...     Subject("P3", "D3", details={"sex": "female", "cognition": "mci"}))
    >>> my_splits = {"sex": (("female",), ("male",)), "cognition": (("hc",), ("mci",), ("ad",), ("mci", "hc"))}
    >>> my_outs = make_subject_splits(subjects=my_subjects, splits=my_splits)
    >>> tuple(my_outs.keys())
    ('sex', 'cognition')
    >>> my_outs["sex"]  # doctest: +NORMALIZE_WHITESPACE
    {Criterion(crit=('female',)):
         (Subject(subject_id='P3', dataset_name='D1', details={'sex': 'female', 'cognition': 'mci'}),
          Subject(subject_id='P1', dataset_name='D2', details={'sex': 'female', 'cognition': 'ad'}),
          Subject(subject_id='P2', dataset_name='D3', details={'sex': 'female', 'cognition': 'hc'}),
          Subject(subject_id='P3', dataset_name='D3', details={'sex': 'female', 'cognition': 'mci'})),
     Criterion(crit=('male',)):
         (Subject(subject_id='P1', dataset_name='D1', details={'sex': 'male', 'cognition': 'hc'}),
          Subject(subject_id='P2', dataset_name='D1', details={'sex': 'male', 'cognition': 'hc'}),
          Subject(subject_id='P2', dataset_name='D2', details={'sex': 'male', 'cognition': 'ad'}),
          Subject(subject_id='P1', dataset_name='D3', details={'sex': 'male', 'cognition': 'mci'}))}
    >>> my_outs["cognition"]  # doctest: +NORMALIZE_WHITESPACE
    {Criterion(crit=('hc',)):
         (Subject(subject_id='P1', dataset_name='D1', details={'sex': 'male', 'cognition': 'hc'}),
          Subject(subject_id='P2', dataset_name='D1', details={'sex': 'male', 'cognition': 'hc'}),
          Subject(subject_id='P2', dataset_name='D3', details={'sex': 'female', 'cognition': 'hc'})),
     Criterion(crit=('mci',)):
         (Subject(subject_id='P3', dataset_name='D1', details={'sex': 'female', 'cognition': 'mci'}),
          Subject(subject_id='P1', dataset_name='D3', details={'sex': 'male', 'cognition': 'mci'}),
          Subject(subject_id='P3', dataset_name='D3', details={'sex': 'female', 'cognition': 'mci'})),
     Criterion(crit=('ad',)):
         (Subject(subject_id='P1', dataset_name='D2', details={'sex': 'female', 'cognition': 'ad'}),
          Subject(subject_id='P2', dataset_name='D2', details={'sex': 'male', 'cognition': 'ad'})),
     Criterion(crit=('mci', 'hc')):
         (Subject(subject_id='P1', dataset_name='D1', details={'sex': 'male', 'cognition': 'hc'}),
          Subject(subject_id='P2', dataset_name='D1', details={'sex': 'male', 'cognition': 'hc'}),
          Subject(subject_id='P3', dataset_name='D1', details={'sex': 'female', 'cognition': 'mci'}),
          Subject(subject_id='P1', dataset_name='D3', details={'sex': 'male', 'cognition': 'mci'}),
          Subject(subject_id='P2', dataset_name='D3', details={'sex': 'female', 'cognition': 'hc'}),
          Subject(subject_id='P3', dataset_name='D3', details={'sex': 'female', 'cognition': 'mci'}))}
    """
    # Loop through all desired splits (e.g. 'sex' and 'education')
    subjects_split: Dict[str, Dict[Criterion, Tuple[Subject, ...]]] = dict()
    for split, criteria in splits.items():
        # Loop through the different criteria (e.g. 'male' and 'female' for sex split)
        subjects_split[split] = dict()
        for criterion in criteria:
            c = criterion.crit if isinstance(criterion, Criterion) else criterion

            # Loop through all subjects to append the ones which satisfy the criterion
            included_subjects = []
            for subject in subjects:
                if split in subject.details and subject.details[split] in c:
                    included_subjects.append(subject)

            # Add the included subjects as criterion satisfied for the split
            subjects_split[split][Criterion(crit=c)] = tuple(included_subjects)

    return subjects_split
