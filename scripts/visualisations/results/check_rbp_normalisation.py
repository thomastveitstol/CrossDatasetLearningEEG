"""
I discovered that the normalise_region_representations key in the config file was sampling from a list with only a
single element, "true". This script confirmed that RBP always used normalisation.

Output (executed 11th of September 2024):
Number of RBP models with normalisation: 543
Number of RBP models without normalisation: 0
Number of interpolation models with normalisation: 263
Number of interpolation models without normalisation: 282
"""
import os

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.results_analysis import get_all_ilodo_runs, get_all_lodo_runs, get_config_file


def main():
    results_dir = get_results_dir()

    runs = (get_all_lodo_runs(results_dir, successful_only=False)
            + get_all_ilodo_runs(results_dir, successful_only=False))
    rbp_normalisation_count_true = 0
    rbp_normalisation_count_false = 0
    interpolation_normalisation_count_true = 0
    interpolation_normalisation_count_false = 0
    for run in runs:
        # Get config file
        config_path = os.path.join(results_dir, run)
        config = get_config_file(results_folder=config_path, preprocessing=False)

        # If this run used RBP, iterate correct counter
        method = config["Varied Numbers of Channels"]["name"]
        if method == "RegionBasedPooling":
            normalisation = config["Varied Numbers of Channels"]["kwargs"]["normalise_region_representations"]

            assert isinstance(normalisation, bool), f"Unexpected type: {type(normalisation)}"
            if normalisation:
                rbp_normalisation_count_true += 1
            else:
                rbp_normalisation_count_false += 1
        elif method == "Interpolation":
            normalisation = config["DL Architecture"]["normalise"]

            assert isinstance(normalisation, bool), f"Unexpected type: {type(normalisation)}"
            if normalisation:
                interpolation_normalisation_count_true += 1
            else:
                interpolation_normalisation_count_false += 1
        else:
            raise ValueError(f"Unexpected method for handling varied numbers of channels: {method}")

    # --------------
    # Print the results
    # --------------
    print(f"Number of RBP models with normalisation: {rbp_normalisation_count_true}")
    print(f"Number of RBP models without normalisation: {rbp_normalisation_count_false}")

    print(f"Number of interpolation models with normalisation: {interpolation_normalisation_count_true}")
    print(f"Number of interpolation models without normalisation: {interpolation_normalisation_count_false}")


if __name__ == "__main__":
    main()
