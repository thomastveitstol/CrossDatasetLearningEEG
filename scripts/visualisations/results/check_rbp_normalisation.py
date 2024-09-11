"""
I discovered that the normalise_region_representations key in the config file was sampling from a list with only a
single element, "true". This script confirmed that RBP always used normalisation.

Output (executed 11th of September 2024):
Number of RBP models with normalisation: 468
Number of RBP models without normalisation: 0
"""
import os

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.results_analysis import get_all_ilodo_runs, get_all_lodo_runs, get_config_file


def main():
    results_dir = get_results_dir()

    runs = get_all_lodo_runs(results_dir) + get_all_ilodo_runs(results_dir)
    normalisation_count_true = 0
    normalisation_count_false = 0
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
                normalisation_count_true += 1
            else:
                normalisation_count_false += 1
        elif method == "Interpolation":
            continue
        else:
            raise ValueError(f"Unexpected method for handling varied numbers of channels: {method}")

    # --------------
    # Print the results
    # --------------
    print(f"Number of RBP models with normalisation: {normalisation_count_true}")
    print(f"Number of RBP models without normalisation: {normalisation_count_false}")


if __name__ == "__main__":
    main()
