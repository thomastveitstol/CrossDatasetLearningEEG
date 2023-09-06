import os

import mne

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, path_method


class Rockhill(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> Rockhill().name
    'rockhill'
    """

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def pre_process(self, eeg_data, *, filtering=None, resample=None, notch_filter=None, avg_reference=False):
        """See the parent class implementation for details. This overriding method excludes non-EEG channels prior to
        pre-processing"""
        # Keep EEG channels only
        non_eeg_channels = [ch_name for ch_name in eeg_data.ch_names if ch_name[:3] == "EXG"]
        eeg_data = eeg_data.pick(picks="eeg", exclude=non_eeg_channels)

        # Run the super method and return
        return super().pre_process(eeg_data, filtering=filtering, resample=resample, notch_filter=notch_filter,
                                   avg_reference=avg_reference)

    # ----------------
    # Methods for different paths
    # ----------------
    @staticmethod
    @path_method
    def _path_to_pd_subject(subject_id, on):
        """
        Get the subject path of a subject with Parkinson's disease

        Parameters
        ----------
        subject_id : str
            Subject ID
        on : bool
            Loads data from the 'on' folder if True, otherwise 'off'
            todo: find out what this means.

        Returns
        -------
        str
            Subject path
        """
        # Create path
        session = "ses-on" if on else "ses-off"
        return os.path.join(subject_id, session, "eeg", f"{subject_id}_{session}_task-rest_eeg.bdf")

    @staticmethod
    @path_method
    def _path_to_hc_subject(subject_id):
        """
        Get the subject path of a healthy control subject

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            Subject path
        """
        return os.path.join(subject_id, "ses-hc", "eeg", f"{subject_id}_ses-hc_task-rest_eeg.bdf")

    @staticmethod
    def _get_subject_status(subject_id):
        """
        Get the subject status (pd or hc)

        Parameters
        ----------
        subject_id : str
            Subject ID

        Returns
        -------
        str
            The status of the subject. 'hc' if healthy control, 'pd' if Parkinson's disease
        """
        # Get and check status
        status = subject_id[4:6]
        assert status in ("hc", "pd"), (f"Expected the status of the subject to be healthy control or parkinsons "
                                        f"disease, but found {status}")

        return status

    # ----------------
    # Methods for loading
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **kwargs):
        # Create path
        status = self._get_subject_status(subject_id)
        if status == "hc":
            subject_path = self._path_to_hc_subject(subject_id)
        elif status == "pd":
            subject_path = self._path_to_pd_subject(subject_id, on=kwargs["on"])
        else:
            raise ValueError("This should never happen. Please contact the developer.")
        path = os.path.join(self.get_mne_path(), subject_path)

        # Make MNE raw object
        return mne.io.read_raw_bdf(input_fname=path, preload=True, verbose=False)

    # ----------------
    # Methods for channel system
    # ----------------
    def channel_name_to_index(self):
        raise NotImplementedError
