import numpy


def chunk_eeg(*, data, k, chunk_duration, delta_t):
    """Function for splitting EEG data into chunks

    (unittest in test folder)
    Parameters
    ----------
    data : numpy.ndarray
        EEG data with shape=(num_subjects, num_channels, num_timesteps)
    k : int
        Number of EEG chunks
    chunk_duration : int
        The duration of the EEG chunks (in number of timesteps)

    delta_t : int:
        time duration between the chunks, in number of timesteps

    Returns
    -------
    tuple[numpy.ndarray, ...]
        The EEG data split into chunks

    """
    # Add the first chunk
    chunk = [data[..., 0:chunk_duration]]

    # Append all the other chunks
    for i in range(1, k):
        i0 = i * (delta_t + chunk_duration)
        chunk.append(data[..., i0:(i0+chunk_duration)])

    return tuple(chunk)
