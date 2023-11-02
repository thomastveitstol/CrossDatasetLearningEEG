

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
    >>> from cdl_eeg.data.data_split import Subject
    >>> my_subjects = (
    ...     Subject("P1", "D1", details={"sex": "male", "age": "old"}),
    ...     Subject("P2", "D1", details={"sex": "male", "age": "old"}),
    ...     Subject("P3", "D1", details={"sex": "female", "age": "young"}),
    ...     Subject("P1", "D2", details={"sex": "female", "age": "old"}),
    ...     Subject("P2", "D2", details={"sex": "male", "age": "young"}),
    ...     Subject("P1", "D3", details={"sex": "male", "age": "old"}),
    ...     Subject("P2", "D3", details={"sex": "female", "age": "old"}),
    ...     Subject("P3", "D3", details={"sex": "female", "age": "young"}))
    >>> my_criteria = {"sex": ("female",), "age": ("young",)}
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
