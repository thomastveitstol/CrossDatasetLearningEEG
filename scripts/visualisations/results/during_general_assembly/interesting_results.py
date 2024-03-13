import os

import seaborn
import yaml
from matplotlib import pyplot

from cdl_eeg.data.paths import get_results_dir


def _get_config_file(folder_name, results_folder, preprocessing):
    file_name = "preprocessing_config.yml" if preprocessing else "config.yml"
    with open(os.path.join(results_folder, folder_name, file_name)) as f:
        config = yaml.safe_load(f)
    return config


def _get_num_montage_splits(config):
    num_montage_splits = 0
    for rbp_design in config["Varied Numbers of Channels"]["RegionBasedPooling"]["RBPDesigns"].values():
        num_montage_splits += len(rbp_design["split_methods"])
    return num_montage_splits


def _get_hyperparameter(config, key_path):
    if key_path == "num_montage_splits":
        return _get_num_montage_splits(config=config)

    hyperparameter = config
    for key in key_path:
        hyperparameter = hyperparameter[key]
    return hyperparameter


def create_data_dict(bad_runs, stable_runs, promising_runs, key_path, results_folder, preprocessing):
    # Initialise
    data_dict = {"Group": [], "Hyperparameter value": []}

    # ---------------
    # Loop through the bad ones
    # ---------------
    for bad in bad_runs:
        # Load config file
        config = _get_config_file(folder_name=bad, results_folder=results_folder, preprocessing=preprocessing)

        # Append values
        data_dict["Group"].append("Bad")
        data_dict["Hyperparameter value"].append(_get_hyperparameter(config=config, key_path=key_path))

    # ---------------
    # Loop through the stable ones
    # ---------------
    for stable in stable_runs:
        # Load config file
        config = _get_config_file(folder_name=stable, results_folder=results_folder, preprocessing=preprocessing)

        # Append values
        data_dict["Group"].append("Stable")
        data_dict["Hyperparameter value"].append(_get_hyperparameter(config=config, key_path=key_path))

    # ---------------
    # Loop through the promising ones
    # ---------------
    for promising in promising_runs:
        # Load config file
        config = _get_config_file(folder_name=promising, results_folder=results_folder, preprocessing=preprocessing)

        # Append values
        data_dict["Group"].append("Promising")
        data_dict["Hyperparameter value"].append(_get_hyperparameter(config=config, key_path=key_path))

    return data_dict


def main():
    # Interesting values: beta_2, learning_rate

    key_path = ("general", "resample")  # "num_montage_splits"
    log_scale = True
    preprocessing = True

    plot_type = "countplot"

    results_folder = os.path.join(get_results_dir(), "ga_runs")

    # -------------------
    # Categorise the runs
    # -------------------
    # Stable AUC curves
    stable = ("debug_sex_cv_experiments_2024-03-13_084931", "debug_sex_cv_experiments_2024-03-13_051344",
              "debug_sex_cv_experiments_2024-03-13_015857", "debug_sex_cv_experiments_2024-03-12_204033",
              "debug_sex_cv_experiments_2024-03-12_115937", "debug_sex_cv_experiments_2024-03-12_064212",
              "debug_sex_cv_experiments_2024-03-12_045559", "debug_sex_cv_experiments_2024-03-12_041804",
              "debug_sex_cv_experiments_2024-03-12_005306", "debug_sex_cv_experiments_2024-03-11_170525",
              "debug_sex_cv_experiments_2024-03-11_082601", "debug_sex_cv_experiments_2024-03-11_071540",
              "debug_sex_cv_experiments_2024-03-11_044622", "debug_sex_cv_experiments_2024-03-11_043719",
              "debug_sex_cv_experiments_2024-03-11_020050", "debug_sex_cv_experiments_2024-03-10_174341",
              "debug_sex_cv_experiments_2024-03-10_104806", "debug_sex_cv_experiments_2024-03-09_230951",
              "debug_sex_cv_experiments_2024-03-09_175730", "debug_sex_cv_experiments_2024-03-09_171635",
              "debug_sex_cv_experiments_2024-03-09_122624", "debug_sex_cv_experiments_2024-03-08_024741")

    # The most promising runs
    promising = ("debug_sex_cv_experiments_2024-03-12_225720", "debug_sex_cv_experiments_2024-03-12_164144",
                 "debug_sex_cv_experiments_2024-03-12_122345",
                 "debug_sex_cv_experiments_2024-03-12_031128",  # this one looked very nice!
                 "debug_sex_cv_experiments_2024-03-11_150102", "debug_sex_cv_experiments_2024-03-11_130721",
                 "debug_sex_cv_experiments_2024-03-11_120442", "debug_sex_cv_experiments_2024-03-11_091408",
                 "debug_sex_cv_experiments_2024-03-11_034959", "debug_sex_cv_experiments_2024-03-10_194235",
                 "debug_sex_cv_experiments_2024-03-10_104806",  # this is also quite stable
                 "debug_sex_cv_experiments_2024-03-09_233910", "debug_sex_cv_experiments_2024-03-08_182107",
                 "debug_sex_cv_experiments_2024-03-08_120136",
                 "debug_sex_cv_experiments_2024-03-08_024741",  # unsure about this one
                 "debug_sex_cv_experiments_2024-03-08_011900", "debug_sex_cv_experiments_2024-03-07_205046")

    # The others
    all_ga_runs = os.listdir(results_folder)
    bads = tuple(run for run in all_ga_runs if run not in stable + promising and run[:5] == "debug")

    # -------------------
    # Get the values per group
    # -------------------
    summary = create_data_dict(bad_runs=bads, stable_runs=stable, promising_runs=promising, key_path=key_path,
                               results_folder=results_folder, preprocessing=preprocessing)

    # -------------------
    # Plotting
    # -------------------
    if plot_type == "histplot":
        seaborn.histplot(summary, hue="Group", x="Hyperparameter value", kde=True, log_scale=log_scale)
    elif plot_type == "countplot":
        seaborn.countplot(summary, hue="Group", x="Hyperparameter value")
    else:
        raise ValueError(f"Plot ype not recognised: {plot_type}")

    pyplot.title(f"Hyperparameter: {key_path}")

    pyplot.show()


if __name__ == "__main__":
    main()
