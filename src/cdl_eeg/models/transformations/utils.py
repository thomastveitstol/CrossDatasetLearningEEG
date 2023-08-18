import numpy


def chunk_eeg(data, k, chuck_duration, delta_t):
    """Function for splitting

    Parameters
    ----------
    data : numpy.ndarray
        EEG data with shape=(num_subjects, num_channels, num_timesteps)
    k : int
        Number of EEG chunks
    chuck_duration : int
        The duration of the EEG chunks (in number of timesteps)

    delta_t : int:
        time duration between the chunks, in number of timesteps

    Returns
    -------
    tuple[numpy.ndarray, ...]
        The EEG data split into chunks

    """
