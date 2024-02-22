def sex_to_int(sex):
    """
    Convert sex to integer. Males are 0s, females are 1s

    Parameters
    ----------
    sex : str

    Returns
    -------
    int
        Either a 0 or a 1

    Examples
    --------
    >>> sex_to_int("m")
    0
    >>> sex_to_int("mAlE")
    0
    >>> sex_to_int("F")
    1
    >>> sex_to_int("FemalE")
    1
    >>> sex_to_int("male")
    0
    >>> sex_to_int("Dude")
    Traceback (most recent call last):
    ...
    ValueError: Expected lower cased 'sex' to be in ('m', 'male', 'f', 'female'), but found 'Dude'
    """
    # Define legal values for males and females
    male = ("m", "male")
    female = ("f", "female")

    # Covert
    if sex.lower() in male:
        return 0
    elif sex.lower() in female:
        return 1
    else:
        raise ValueError(f"Expected lower cased 'sex' to be in {male + female}, but found '{sex}'")
