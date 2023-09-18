"""
Contains only a function for returning a specified MTS module
"""
from cdl_eeg.models.mts_modules.inception_time import InceptionTime


def get_mts_module(mts_module_name, **kwargs):
    """
    Function for getting a specified MTS module

    Parameters
    ----------
    mts_module_name : str
        MTS Module
    kwargs
        Key word arguments, which depends on the selected MTS module

    Returns
    -------
    cdl_eeg.models.mts_modules.mts_module_base.MTSModuleBase

    Examples
    --------
    >>> _ = get_mts_module("InceptionTime", in_channels=5, num_classes=3)
    >>> get_mts_module("NotAnMTSModule")  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The MTS module 'NotAnMTSModule' was not recognised. Please select among the following:
    ('InceptionTime',...)
    """
    # All available MTS modules must be included here
    available_mts_modules = (InceptionTime,)

    # Loop through and select the correct one
    for mts_module in available_mts_modules:
        if mts_module_name == mts_module.__name__:
            return mts_module(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The MTS module '{mts_module_name}' was not recognised. Please select among the following: "
                     f"{tuple(mts_module.__name__ for mts_module in available_mts_modules)}")
