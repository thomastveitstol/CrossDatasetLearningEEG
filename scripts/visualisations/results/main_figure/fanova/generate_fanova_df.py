import itertools
import os
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy
import pandas
import yaml
from ConfigSpace import ConfigurationSpace
from fanova import fANOVA

from cdl_eeg.data.analysis.results_analysis import add_hp_configurations_to_dataframe, get_fanova_hp_distributions, \
    combine_conditions, get_fanova_encoding, higher_is_better
from cdl_eeg.data.paths import get_results_dir


def main():
    numpy.random.seed(1)

    # ----------------
    # A few design choices for the analysis
    # ----------------
    experiment_type = "lodo"
    hps = ("DL architecture", "Band-pass filter", "Normalisation", "Learning rate")
    percentiles = (0, 50, 75, 90, 95)
    selection_metric = "mae"
    target_metric = selection_metric
    fanova_kwargs = {"n_trees": 64, "max_depth": 64}
    decimals = 5

    results_path = Path(get_results_dir())
    save_path = Path(os.path.dirname(__file__))

    # Conditions and columns and rows
    if experiment_type.lower() == "lodo":
        conditions = {"Source dataset": ("Pooled",)}
    elif experiment_type.lower() == "lodi":
        conditions = {"Target dataset": ("Pooled",)}
    else:
        raise ValueError(f"Unexpected experiment type: {experiment_type}")

    # ----------------
    # Get the data
    # ----------------
    # Load the results df
    _root_path = os.path.dirname(os.path.dirname(__file__))
    path = Path(_root_path) / f"all_test_results_selection_metric_{selection_metric}.csv"
    df = pandas.read_csv(path)
    df.fillna(0.0, inplace=True)

    # Extract subset
    if conditions:
        _current_conditions = {col: values for col, values in conditions.items() if col in df.columns}
        combined_conditions = combine_conditions(df=df, conditions=_current_conditions)
        df = df[combined_conditions]

    # ----------------
    # Create fANOVA objects
    # ----------------
    # Add HPCs to dataframe
    df = add_hp_configurations_to_dataframe(df=df, hps=hps, results_dir=results_path)

    # Do numerical encoding as required by fANOVA
    df.replace(get_fanova_encoding(), inplace=True)

    # Get the config file with HP distributions
    curr_path = Path(__file__)
    scripts_folder_path = Path(*curr_path.parts[:curr_path.parts.index("scripts") + 1])
    path_to_hpd_config = (scripts_folder_path / "models" / "training" / "config_files"
                          / "hyperparameter_random_search.yml")
    with open(path_to_hpd_config, "r") as file:
        hpd_config = yaml.safe_load(file)

    # Get the distibutions and make config space
    config_space = ConfigurationSpace(get_fanova_hp_distributions(hps, hpd_config=hpd_config))

    # Create fanova objects per
    source_datasets = set(df["Source dataset"])
    target_datasets = set(df["Target dataset"])
    fanovas: Dict[Tuple[str, str], fANOVA] = dict()
    for source_dataset, target_dataset in itertools.product(source_datasets, target_datasets):
        if source_dataset == target_dataset:
            continue

        # Get the subset
        subset_df = df[(df["Source dataset"] == source_dataset) & (df["Target dataset"] == target_dataset)]
        subset_df = subset_df.reset_index(inplace=False)

        # Create fANOVA object
        fanovas[(source_dataset, target_dataset)] = fANOVA(X=subset_df[list(hps)], Y=subset_df[target_metric],
                                                           config_space=config_space, **fanova_kwargs)

    print(config_space)

    # ----------------
    # fANOVA analysis
    # ----------------
    # The fanova package uses numpy.float, which is deprecated. The error message says that replacing with 'float' is
    # safe
    numpy.float = float

    print("Computing marginals...")
    # Marginal importance
    marginal_importance: Dict[str, List[Any]] = {"Target dataset": [], "Source dataset": [], "Percentile": [],
                                                 **{f"{hp_name} (mean)": [] for hp_name in hps},
                                                 **{f"{hp_name} (std)": [] for hp_name in hps}}
    for (source_dataset, target_dataset), fanova_object in fanovas.items():
        for percentile in percentiles:
            # Compute cutoffs
            subset_df = df[(df["Source dataset"] == source_dataset) & (df["Target dataset"] == target_dataset)]
            if higher_is_better(target_metric):
                lower_cutoff = numpy.percentile(subset_df[target_metric], percentile)
                upper_cutoff = numpy.inf
            else:
                lower_cutoff = -numpy.inf
                upper_cutoff = numpy.percentile(subset_df[target_metric], 100 - percentile)
            fanova_object.set_cutoffs(cutoffs=(lower_cutoff, upper_cutoff))

            # Loop though all desired HPs
            for hp_name in hps:
                summary = fanova_object.quantify_importance(dims=(hp_name,))[(hp_name,)]
                importance = summary["individual importance"]
                std = summary["individual std"]

                # Add to marginal importance
                marginal_importance[f"{hp_name} (mean)"].append(importance)
                marginal_importance[f"{hp_name} (std)"].append(std)

            # Add the rest of the info to marginal importance results
            marginal_importance["Target dataset"].append(target_dataset)
            marginal_importance["Source dataset"].append(source_dataset)
            marginal_importance["Percentile"].append(percentile)

    # Save the results
    pandas.DataFrame(marginal_importance).round(decimals).to_csv(
        save_path / f"marginal_importance_{experiment_type}_{selection_metric}_{target_metric}.csv"
    )

if __name__ == "__main__":
    main()
