"""
Contains only a function for returning a specified region split
"""
from cdl_eeg.models.region_based_pooling.montage_splits.voronoi_split import VoronoiSplit


def get_region_split(split_method, **kwargs):
    """
    Function for getting the specified region split.

    Parameters
    ----------
    split_method : str
        Name of the region split class name
    kwargs
        Key word arguments, which depends on the selected region split

    Returns
    -------
    cdl_eeg.models.region_based_pooling.region_splits.region_split_base.RegionSplitBase
        The region splits

    Examples
    --------
    >>> _ = get_region_split("VoronoiSplit", num_points=7, x_min=0, x_max=1, y_min=0, y_max=1)
    >>> get_region_split("NotASplitMethod")  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The split method 'NotASplitMethod' was not recognised. Please select among the following:
    ('VoronoiSplit',...)
    """
    # All available regions splits must be included here
    available_region_splits = (VoronoiSplit,)

    # Loop through and select the correct one
    for region_split in available_region_splits:
        if split_method == region_split.__name__:
            return region_split(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The split method '{split_method}' was not recognised. Please select among the following: "
                     f"{tuple(pooling_module.__name__ for pooling_module in available_region_splits)}")
