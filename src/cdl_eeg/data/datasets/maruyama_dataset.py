import os
from typing import Tuple

import mne

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, path_method


class Maruyama(EEGDatasetBase):
    """
    https://zenodo.org/record/3630583
    https://doi.org/10.1016/j.neures.2020.01.013
    https://www.sciencedirect.com/science/article/pii/S0168010219306765?via%3Dihub

    Examples:
    ----------
    >>> Maruyama().name
    'maruyama'
    """

    __slots__ = ()

    # ----------------
    # Methods for different paths
    # ----------------
    def get_subject_ids(self) -> Tuple[str, ...]:
        # Extract by folder names
        healthy = tuple(os.listdir(os.path.join(self.get_mne_path(), "Raw EEG", "Healthy participant EEG")))
        patients = tuple(os.listdir(os.path.join(self.get_mne_path(), "Raw EEG", "Patient EEG")))

        # Merge and return
        return healthy + patients

    @staticmethod
    def _get_subject_status(subject_id):
        """
        Get the status of a subject ('healthy' or 'patient')
        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            The status of the subejct, either 'healthy' or 'patient'
        """
        if subject_id[0] == "H":
            return "healthy"
        elif subject_id[0] == "P":
            return "patient"
        else:
            raise ValueError(f"Expected subject name to start with 'H' or 'P' (indicating healthy or patient), but "
                             f"received the subject ID {subject_id}")

    @staticmethod
    @path_method
    def _path_to_healthy_subject(subject_id):
        """
        Get the subject path of a healthy subject

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            Subject path
        """
        return os.path.join("Raw EEG", "Healthy participant EEG", subject_id, f"{subject_id}.dat")

    @staticmethod
    @path_method
    def _path_to_patient_subject(subject_id):
        """
        Get the subject path of a patient. Returning with .vhdr (header file), as that is the appropriate argument to
        the MNE loading function

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            Subject path
        """
        return os.path.join("Raw EEG", "Patient EEG", subject_id, f"{subject_id}.vhdr")

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **kwargs):
        # Create path
        status = self._get_subject_status(subject_id)
        if status == "healthy":
            subject_path = self._path_to_healthy_subject(subject_id)
        elif status == "patient":
            subject_path = self._path_to_patient_subject(subject_id)
        else:
            raise ValueError("This should never happen. Please contact the developer.")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Load the MNE file (healthy subjects have a .dat file, whereas patients have a .eeg file)
        if status == "healthy":
            # todo: does not work, as I get FileNotFoundError. I happens because it is not on Persyst format. See
            #  https://mne.discourse.group/t/read-and-analysis-eeg-from-dat-file/3810/5 for fixing
            return mne.io.read_raw_persyst(path, preload=True, verbose=False)
        elif status == "patient":
            return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        else:
            raise ValueError("This should never happen. Please contact the developer.")

    # ----------------
    # Channel system
    # ----------------
    def channel_name_to_index(self):
        raise NotImplementedError
