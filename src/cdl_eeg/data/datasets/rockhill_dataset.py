import os

import mne
import numpy
import pandas

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, path_method, target_method


class Rockhill(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> Rockhill().name
    'rockhill'
    >>> Rockhill.get_available_targets()
    ('age', 'mmse', 'naart', 'parkinsons')
    >>> len(Rockhill().get_subject_ids())
    31
    """

    __slots__ = ()

    _channel_names = ('Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1',
                      'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
                      'Fz', 'Cz')  # Removed EXG channels, as these channel names are supposed to match numpy arrays

    def pre_process(self, eeg_data, *, filtering=None, resample=None, notch_filter=None, avg_reference=False,
                    interpolation=None, remove_above_std, excluded_channels="EXG"):
        """See the parent class implementation for details. This overriding method excludes non-EEG channels prior to
        pre-processing by default"""
        if excluded_channels == "EXG":
            # Keep EEG channels only
            excluded_channels = tuple(ch_name for ch_name in eeg_data.ch_names if ch_name[:3] == "EXG")

        # Run the super method and return
        return super().pre_process(eeg_data, filtering=filtering, resample=resample, notch_filter=notch_filter,
                                   avg_reference=avg_reference, excluded_channels=excluded_channels,
                                   remove_above_std=remove_above_std, interpolation=interpolation)

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
        Get the subject status (pd or hc)  todo: consider using Enum instead of string

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
    # Targets  todo: they are very similar...
    # ----------------
    @target_method
    def parkinsons(self, subject_ids):
        # Loop though subject IDs
        targets = []
        for sub_id in subject_ids:
            # Set the target to 0 if healthy, 1 if PD
            status = self._get_subject_status(sub_id)
            if status == "hc":
                targets.append(0)
            elif status == "pd":
                targets.append(1)
            else:
                raise ValueError("This should never happens, please contact the developer")

        # Convert to numpy array and return
        return numpy.array(targets)

    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["age"])}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def mmse(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["MMSE"])}

        # Extract the MMSE score of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    @target_method
    def naart(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path(), sep="\t")

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["participant_id"], df["NAART"])}

        # Extract the NAART score of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])

    # ----------------
    # Methods for channel system
    # ----------------
    def _get_template_electrode_positions(self):
        # todo: This looks really bad when plotting...
        # Following the international 10-20 system according to the README file. Thus using MNE default
        montage = mne.channels.make_standard_montage("standard_1020")
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        # The _channel_names were checked for two subjects and manually controlled
        # todo: Make unittest to verify that the indexing is true for all subjects in the dataset
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}
