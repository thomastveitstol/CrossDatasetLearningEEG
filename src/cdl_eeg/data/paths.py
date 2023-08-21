"""
Functions for getting the path to different files related to the data and results
"""


def get_raw_data_storage_path():
    """
    Get the path to where the newly downloaded data is (supposed to be) stored.

    Note that this will only work for me (Thomas). I could not store the datasets inside this Python-project, as they
    require too much memory
    Returns
    -------
    str
        The path to where the data is stored, or to be stored (e.g. in the scripts for downloading)
    """
    return "/media/thomas/AI-Mind - Anonymised data"
