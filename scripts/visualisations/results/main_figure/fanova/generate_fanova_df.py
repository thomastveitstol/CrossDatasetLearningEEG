import itertools
import os
from collections import OrderedDict
from pathlib import Path
import random
from typing import Dict, Tuple, Any, List

import numpy
import pandas
import yaml
from ConfigSpace import ConfigurationSpace
from fanova import fANOVA

from cdl_eeg.data.analysis.results_analysis import add_hp_configurations_to_dataframe, get_fanova_hp_distributions, \
    combine_conditions, get_fanova_encoding, higher_is_better
from cdl_eeg.data.paths import get_results_dir


# ------------
# Updated classes
# ------------
class UpdatedFANOVA(fANOVA):
    """
    Just need make some changes to a method, original fANOVA (parent class) is found at
    https://github.com/automl/fanova/blob/master/fanova/fanova.py
    """

    def get_most_important_pairwise_marginals(self, params=None, n=None):
        """
        Similar to base class, but returns all pairwise marginals and returns the standard
        deviations too.

        Most is taken from the original function, see:
        https://github.com/automl/fanova/blob/master/fanova/fanova.py
        """
        tot_imp_dict = OrderedDict()
        pairwise_marginals = []
        if params is None:
            dimensions = range(self.n_dims)
        else:
            if type(params[0]) == str:
                idx = []
                for i, param in enumerate(params):
                    idx.append(self.cs.get_idx_by_hyperparameter_name(param))
                dimensions = idx

            else:
                dimensions = params
        # pairs = it.combinations(dimensions,2)
        pairs = [x for x in itertools.combinations(dimensions, 2)]
        if params:
            n = len(list(pairs))

        try:
            from progressbar import progressbar  # type: ignore
            hp_pair_loop = progressbar(pairs, redirect_stdout=True, prefix="HP pairs ")
        except ImportError:
            hp_pair_loop = pairs
        for combi in hp_pair_loop:
            pairwise_marginal_performance = self.quantify_importance(combi)
            tot_imp = pairwise_marginal_performance[combi]['individual importance']  # Importance
            std = pairwise_marginal_performance[combi]['individual std']  # std (added)
            combi_names = [self.cs_params[combi[0]].name, self.cs_params[combi[1]].name]  # HP names
            pairwise_marginals.append((tot_imp, std, combi_names[0], combi_names[1]))

        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)

        if n is None:
            for marginal, std, p1, p2 in pairwise_marginal_performance:
                tot_imp_dict[(p1, p2)] = marginal, std
        else:
            for marginal, std, p1, p2 in pairwise_marginal_performance[:n]:
                tot_imp_dict[(p1, p2)] = marginal, std
        self._dict = True

        return tot_imp_dict


# ------------
# Functions for generating dataframes
# ------------
def _generate_marginals_df(df, *, decimals, experiment_type, fanovas, hps, percentiles, save_path, selection_metric,
                           target_metric):
    marginal_importance: Dict[str, List[Any]] = {"Target dataset": [], "Source dataset": [], "Percentile": [],
                                                 **{f"{hp_name} (mean)": [] for hp_name in hps},
                                                 **{f"{hp_name} (std)": [] for hp_name in hps}}

    try:
        from progressbar import progressbar  # type: ignore
        fanova_loop = progressbar(fanovas.items(), redirect_stdout=True, prefix="Source/target combination ")
    except ImportError:
        fanova_loop = fanovas.items()
    for (source_dataset, target_dataset), fanova_object in fanova_loop:
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


def _generate_pairwise_df(df, *, decimals, experiment_type, fanovas, num_pairwise_marginals, percentiles, save_path,
                          selection_metric, target_metric):
    pairwise_marginals: Dict[str, List[Any]] = {"Target dataset": [], "Source dataset": [], "Percentile": [],
                                                "Rank": [], "HP1": [], "HP2": [], "Importance": [], "Std": []}

    try:
        from progressbar import progressbar  # type: ignore
        fanova_loop = progressbar(fanovas.items(), redirect_stdout=True, prefix="Source/target combination ")
    except ImportError:
        fanova_loop = fanovas.items()
    for (source_dataset, target_dataset), fanova_object in fanova_loop:
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

            # Compute interactions
            hp_interaction_ranking = fanova_object.get_most_important_pairwise_marginals(n=num_pairwise_marginals)
            for rank, ((hp_1, hp_2), (importance, std)) in enumerate(hp_interaction_ranking.items()):
                # Add results. The HPs are ranked naturally
                pairwise_marginals["HP1"].append(hp_1)
                pairwise_marginals["HP2"].append(hp_2)
                pairwise_marginals["Rank"].append(rank)
                pairwise_marginals["Target dataset"].append(target_dataset)
                pairwise_marginals["Source dataset"].append(source_dataset)
                pairwise_marginals["Percentile"].append(percentile)
                pairwise_marginals["Importance"].append(importance)
                pairwise_marginals["Std"].append(std)

    # Save the results
    pandas.DataFrame(pairwise_marginals).round(decimals).to_csv(
        save_path / f"pairwise_importance_{experiment_type}_{selection_metric}_{target_metric}.csv"
    )


def _generate_dataframes(*, conditions, decimals, experiment_type, fanova_kwargs, hps, num_pairwise_marginals,
                         percentiles, results_path, save_path, selection_metric, skip_pairwise_importance,
                         target_metric):
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
    config_space = ConfigurationSpace(get_fanova_hp_distributions(hps, hpd_config=hpd_config,
                                                                  experiment_type=experiment_type))
    print(config_space)

    # Create fanova objects per
    source_datasets = set(df["Source dataset"])
    target_datasets = set(df["Target dataset"])
    fanovas: Dict[Tuple[str, str], UpdatedFANOVA] = dict()
    for source_dataset, target_dataset in itertools.product(source_datasets, target_datasets):
        if source_dataset == target_dataset:
            continue

        # Get the subset
        subset_df = df[(df["Source dataset"] == source_dataset) & (df["Target dataset"] == target_dataset)]
        subset_df = subset_df.reset_index(inplace=False)

        # Create fANOVA object
        fanovas[(source_dataset, target_dataset)] = UpdatedFANOVA(
            X=subset_df[list(hps)], Y=subset_df[target_metric], config_space=config_space, **fanova_kwargs
        )

    # ----------------
    # fANOVA analysis
    # ----------------
    # The fanova package uses numpy.float, which is deprecated. The error message says that replacing with 'float' is
    # safe
    numpy.float = float

    print("Computing marginals...")
    # Compute marginal importance
    _generate_marginals_df(
        df, decimals=decimals, experiment_type=experiment_type, fanovas=fanovas, hps=hps, percentiles=percentiles,
        save_path=save_path, selection_metric=selection_metric, target_metric=target_metric
    )

    # (Maybe) compute interactions
    if not skip_pairwise_importance:
        print("Computing pairwise importance...")
        _generate_pairwise_df(
            df, decimals=decimals, experiment_type=experiment_type, fanovas=fanovas,
            num_pairwise_marginals=num_pairwise_marginals, percentiles=percentiles, save_path=save_path,
            selection_metric=selection_metric, target_metric=target_metric
        )

def main():
    meaning_of_life = 42

    random.seed(meaning_of_life)
    numpy.random.seed(meaning_of_life)

    # ----------------
    # A few design choices for the analysis
    # ----------------
    skip_pairwise_importance = False
    experiment_types = ("lodo", "lodi")
    percentiles = (0, 50, 75, 90, 95)
    selection_metric = "pearson_r"
    target_metric = selection_metric
    fanova_kwargs = {"n_trees": 64, "max_depth": 64}
    decimals = 5
    num_pairwise_marginals = None

    results_path = Path(get_results_dir())
    save_path = Path(os.path.dirname(__file__))

    for experiment_type in experiment_types:
        hps = ("DL architecture", "Band-pass filter", "Domain Adaptation", "Normalisation", "Learning rate",
               r"$\beta_1$", r"$\beta_2$", r"$\epsilon$", "Spatial method", "Sampling frequency", "Input length",
               "Autoreject", "Loss")
        if experiment_type.lower() == "lodo":
            hps += (r"Weighted loss ($\tau$)",)

        # Conditions and columns and rows
        if experiment_type.lower() == "lodo":
            conditions = {"Source dataset": ("Pooled",)}
        elif experiment_type.lower() == "lodi":
            conditions = {"Target dataset": ("Pooled",)}
            # conditions = {"Source dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang")}
        else:
            raise ValueError(f"Unexpected experiment type: {experiment_type}")

        _generate_dataframes(
            conditions=conditions, decimals=decimals, experiment_type=experiment_type, fanova_kwargs=fanova_kwargs,
            hps=hps, num_pairwise_marginals=num_pairwise_marginals, percentiles=percentiles, results_path=results_path,
            save_path=save_path, selection_metric=selection_metric, skip_pairwise_importance=skip_pairwise_importance,
            target_metric=target_metric
        )


if __name__ == "__main__":
    main()
