import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase


class MPILemon(EEGDatasetBase):
    """
    Examples:
    ----------
    >>> MPILemon().name
    'mpi_lemon'
    """

    __slots__ = ()

    def __init__(self):
        super().__init__(name="mpi_lemon")

    # ----------------
    # Loading methods
    # ----------------
    def _load_single_raw_mne_object(self, subject_id, **kwargs):
        raise NotImplementedError

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

                # Download only eyes closed .set and .fdt files. If, in the futre, we want to include eyes open, this is
                # where to change the code
                if "_EC.set" == file_path[-7:] or "_EC.fdt" in file_path[-7:]:
                    # Get subject ID and file type from the folder name
                    subject_id = file_path.split("/")[-2]
                    file_type = file_path.split(".")[-1]  # either .set or .fdt

                    # (Maybe) make folder. The .set and .fdt of a single subject must be placed in the same folder
                    path = os.path.join(to_path, subject_id)
                    if not os.path.isdir(path):
                        os.mkdir(path)

                    # Download
                    s3_bucket.download_file(file_path, os.path.join(path, f"{subject_id}.{file_type}"))

    # ----------------
    # Channel system
    # ----------------
    def channel_name_to_index(self):
        raise NotImplementedError
