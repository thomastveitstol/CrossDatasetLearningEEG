import os

import mne
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, path_method


class VanHees(EEGDatasetBase):
    """
    https://www.biorxiv.org/content/10.1101/324954v1
    https://zenodo.org/record/1252141

    Examples:
    ----------
    >>> VanHees().name
    'van_hees'
    """

    __slots__ = ()
    _channel_names = "AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"
    _sampling_freq = 128  # See Sec. 2.4 in the paper

    # ----------------
    # Methods for different paths
    # ----------------
    @staticmethod
    def _get_subject_location(subject_id):
        """
        Get the country of a subject ('ni' if Nigeria, 'gb' if Guinea-Bissau)

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            'ni' if the file is from Nigeria, 'gb' if Guinea-Bissau

        Examples
        -------
        >>> VanHees._get_subject_location("signal-5.csv.gz")
        'gb'
        >>> VanHees._get_subject_location("signal-5-1.csv.gz")
        'ni'
        >>> VanHees._get_subject_location("signal_5_1.csv.gz")  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: Could not infere the country of the subject 'signal_5_1.csv.gz': Expected one or two hyphens
            in the subject ID name, but received 0
        """
        # An observation on the names of the .csv.gz files, is that the Nigerian files have two hyphens, whereas the
        # files from Guinea-Bissau only have one. Using this to infer the location
        num_hyphens = subject_id.count("-")

        # Return based on number of hyphens
        if num_hyphens == 1:
            return "gb"
        elif num_hyphens == 2:
            return "ni"
        else:
            raise ValueError(f"Could not infere the country of the subject '{subject_id}': Expected one or two hyphens "
                             f"in the subject ID name, but received {num_hyphens}")

    @staticmethod
    @path_method
    def _path_to_nigerian_subject(subject_id):
        """
        Get the subject path of a Nigerian subject

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            Subject path
        """
        return os.path.join("EEGs_Nigeria", f"{subject_id}.csv.gz")

    @staticmethod
    @path_method
    def _path_to_guinea_bissau_subject(subject_id):
        """
        Get the subject path of a Nigerian subject

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            Subject path
        """
        return os.path.join("EEGs_Guinea-Bissau", f"{subject_id}.csv.gz")

    # ----------------
    # Methods for loading
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **_):
        # Create path
        country = self._get_subject_location(subject_id)
        if country == "gb":
            subject_path = self._path_to_guinea_bissau_subject(subject_id)
        elif country == "ni":
            subject_path = self._path_to_nigerian_subject(subject_id)
        else:
            raise ValueError("This should never happen. Please contact the developer.")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Read the .csv.gz file and keep only the channels. Also, convert to numpy and transpose to obtain
        # shape=(channels, time_steps). Finally, divide by a million as the original unit is off
        # TODO: I don't really know what all other columns are. E.g., what is CQ_{channel_name}?
        data = numpy.transpose(_read_csv_gz_file(path)[list(self._channel_names)].to_numpy()) / 1_000_000

        # Create MNE object and return
        info = mne.create_info(ch_names=list(self._channel_names), sfreq=self._sampling_freq, ch_types="eeg")
        return mne.io.RawArray(data, info=info, verbose=False)

    # ----------------
    # Channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # They used international 10-20 according to Sec. 2.4 in the paper
        montage = mne.channels.make_standard_montage("standard_1020")
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}


# ----------------
# Functions
# ----------------
def _read_csv_gz_file(path):
    """
    Read a .csv.gz file using pandas

    Parameters
    ----------
    path : str
        Path to the .csv.gz file

    Returns
    -------
    pandas.DataFrame
    """
    return pandas.read_csv(path, compression="gzip")
