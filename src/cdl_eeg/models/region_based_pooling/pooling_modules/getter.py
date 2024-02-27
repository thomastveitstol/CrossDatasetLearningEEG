"""
Simply a function for returning a specified pooling module

todo: This cannot be the best solution... maybe something with __all__ in init file?
"""
from cdl_eeg.models.region_based_pooling.pooling_modules.head_region import MultiMSSharedRocketHeadRegion
from cdl_eeg.models.region_based_pooling.pooling_modules.mean import SingleCSMean
from cdl_eeg.models.region_based_pooling.pooling_modules.univariate_rocket import MultiCSSharedRocket


def get_pooling_module(pooling_method, **kwargs):
    """
    Function for getting the specified pooling method.

    Parameters
    ----------
    pooling_method : str
        Pooling module
    kwargs
        Key word arguments, which depends on the selected pooling module

    Returns
    -------
    cdl_eeg.models.region_based_pooling.pooling_modules.pooling_base.PoolingModuleBase

    Examples
    --------
    >>> _ = get_pooling_module("SingleCSMean")
    >>> get_pooling_module("NotAPoolingModule")  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The pooling method 'NotAPoolingModule' was not recognised. Please select among the following:
    ('SingleCSMean',...)
    """
    # All available pooling modules must be included here
    available_pooling_modules = (SingleCSMean, MultiCSSharedRocket, MultiMSSharedRocketHeadRegion)

    # Loop through and select the correct one
    for pooling_module in available_pooling_modules:
        if pooling_method == pooling_module.__name__:
            # todo: why does mypy complain in the return line?
            return pooling_module(**kwargs)  # type: ignore[call-arg]

    # If no match, an error is raised
    raise ValueError(f"The pooling method '{pooling_method}' was not recognised. Please select among the following: "
                     f"{tuple(pooling_module.__name__ for pooling_module in available_pooling_modules)}")
