"""
Script for getting the number of subject per dataset and preprocessing version

Conclusion: All preprocessed versions contained the same (and expected) number of subjects
"""
import os
from typing import Dict, List

from cdl_eeg.data.paths import get_numpy_data_storage_path


def main():
    root_path = get_numpy_data_storage_path()
    preprocessed_folders = ("preprocessed_2024-05-13_173548", "preprocessed_2024-04-29_164550")  # These were used

    # ----------------
    # Get all dataset sizes of the preprocessed versions
    # ----------------
    # Need both 5 and 10 seconds, it was stored a little strangely
    dataset_sizes: Dict[str, Dict[int, List[str]]] = {}
    for preprocessed_folder in preprocessed_folders:
        preprocessed_root_folder = os.path.join(root_path, preprocessed_folder)
        for preprocessed_version in os.listdir(preprocessed_root_folder):
            # Everything are folders containing preprocessed versions of the data, and a config.yml file
            if preprocessed_version == "config.yml":
                continue

            # Loop through all datasets
            preprocessed_version_path = os.path.join(preprocessed_root_folder, preprocessed_version)
            for dataset_name in os.listdir(preprocessed_version_path):
                # Compute number of subjects
                num_subjects = len(os.listdir(os.path.join(preprocessed_version_path, dataset_name)))

                # Add to dict
                if dataset_name not in dataset_sizes:
                    dataset_sizes[dataset_name] = {}
                if num_subjects not in dataset_sizes[dataset_name]:
                    dataset_sizes[dataset_name][num_subjects] = []
                dataset_sizes[dataset_name][num_subjects].append(preprocessed_version)

    # ----------------
    # Print results
    # ----------------
    for dataset_name, sizes in dataset_sizes.items():
        print(f"\n--- {dataset_name} ---")
        for size, versions in sizes.items():
            print(f"{size} (N={len(versions)}): {versions}")


if __name__ == "__main__":
    main()
