import os
from typing import Tuple

import boto3
import numpy
import pandas
from botocore import UNSIGNED
from botocore.client import Config
import mne

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase, target_method


class MPILemon(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> MPILemon().name
    'mpi_lemon'
    >>> len(MPILemon().get_subject_ids())
    203
    """

    __slots__ = ()

    _channel_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4",
                      "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2",
                      "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C5",
                      "C1", "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7", "PO3", "POz",
                      "PO4", "PO8")  # TODO: This is inconsistent!!!

    def __init__(self):
        super().__init__(name="mpi_lemon")

    # ----------------
    # Loading methods
    # ----------------
    def get_participants_tsv_path(self):
        # todo: the method name says tsv, but it is a csv file...
        return os.path.join(self.get_mne_path(), "Participants_MPILMBB_LEMON.csv")

    def get_subject_ids(self) -> Tuple[str, ...]:
        # TODO: I think that MPI Lemon has a participants.tsv file as well
        return tuple(os.listdir(self.get_mne_path()))

    def _load_single_raw_mne_object(self, subject_id, **_):
        # Create path
        path = os.path.join(self.get_mne_path(), subject_id, f"{subject_id}.set")

        # Load MNE object and return
        return mne.io.read_raw_eeglab(path, preload=True, verbose=False)

    def download(self, to_path=None):
        """
        Method for downloading the MPI Lemon dataset, eyes closed EEG data only

        Created by Mats Tveter and Thomas Tveitstøl
        Parameters
        ----------
        to_path : str, optional
            Path of where to store the data. Defaults to None (recommended), as it will then download to the expected
            path provided by the .get_mne_path() method.

        Returns
        -------
        None
        """
        # Make root directory
        to_path = self.get_mne_path() if to_path is None else to_path
        os.mkdir(to_path)

        # MPI Lemon specifications
        bucket = 'fcp-indi'
        prefix = "data/Projects/INDI/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed"

        s3client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        # Creating buckets needed for downloading the files
        s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        s3_bucket = s3_resource.Bucket(bucket)

        # Paginator is need because the amount of files exceeds the boto3.client possible maxkeys
        paginator = s3client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        # Looping through the content of the bucket
        for page in pages:
            for obj in page['Contents']:
                # Get the path of the .set or .fdt path
                # e.g. data/Projects/.../EEG_Preprocessed/sub-032514/sub-032514_EO.set
                file_path = obj['Key']

                # Download only eyes closed .set and .fdt files. If, in the future, we want to include eyes open, this
                # is where to change the code
                if "_EC.set" == file_path[-7:] or "_EC.fdt" in file_path[-7:]:
                    # Get subject ID and file type from the folder name
                    subject_id = file_path.split("/")[-2]
                    file_type = file_path.split(".")[-1]  # either .set or .fdt

                    # (Maybe) make folder. The .set and .fdt of a single subject must be placed in the same folder (a
                    # requirement from MNE when loading)
                    path = os.path.join(to_path, subject_id)
                    if not os.path.isdir(path):
                        os.mkdir(path)

                    # Download
                    s3_bucket.download_file(file_path, os.path.join(path, f"{subject_id}.{file_type}"))

        # Participants file
        s3_bucket.download_file("data/Projects/INDI/MPI-LEMON/Participants_MPILMBB_LEMON.csv",
                                os.path.join(to_path, "Participants_MPILMBB_LEMON.csv"))

    # ----------------
    # Channel system
    # ----------------
    def _get_electrode_positions(self, subject_id=None):
        # -----------------
        # Get electrodes from .fdt file
        # -----------------
        # Create path
        path = os.path.join(self.get_mne_path(), subject_id, f"{subject_id}.set")

        # Load MNE object, but not the data. The 'info' object should contain information from the .tsv file
        ch_names = mne.io.read_raw_eeglab(path, preload=False, verbose=False).info["ch_names"]

        # Use MNE montage
        montage = mne.channels.make_standard_montage("standard_1020")
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in ch_names}

    def _get_template_electrode_positions(self):
        # TODO: verify that it is the international 10-20 system
        # TODO: channel present in the data is inconsistent!!!
        montage = mne.channels.make_standard_montage("standard_1020")
        channel_positions = montage.get_positions()["ch_pos"]

        # Return dict with channel positions, keeping only the ones in the data
        return {ch_name: tuple(pos) for ch_name, pos in channel_positions.items() if ch_name in self._channel_names}

    def channel_name_to_index(self):
        # todo: make tests
        # TODO: channel present in the data is inconsistent!!!
        return {ch_name: i for i, ch_name in enumerate(self._channel_names)}

    # ----------------
    # Target methods
    # ----------------
    @target_method
    def age(self, subject_ids):
        # Read the .tsv file
        df = pandas.read_csv(self.get_participants_tsv_path())

        # Setting age to the mean of lower and upper bound of the interval
        age_intervals = df["Age"]
        lower = numpy.array([int(age_interval.split("-")[0]) for age_interval in age_intervals])
        upper = numpy.array([int(age_interval.split("-")[1]) for age_interval in age_intervals])

        mean_age = (upper + lower) / 2

        # Convert to dict
        sub_id_to_age = {name: age for name, age in zip(df["Unnamed: 0"], mean_age)}

        # Extract the ages of the subjects, in the same order as the input argument
        return numpy.array([sub_id_to_age[sub_id] for sub_id in subject_ids])
