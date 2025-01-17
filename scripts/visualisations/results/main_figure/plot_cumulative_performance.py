import os
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import combine_conditions, get_rename_mapping, \
    add_hp_configurations_to_dataframe
from cdl_eeg.data.paths import get_results_dir


def main():
    # ------------
    # Some choices
    # ------------
    target_metric = "mae"
    conditions = {"Target dataset": ("Pooled",)}
    renamed_df_mapping = get_rename_mapping()
    selection_metrics = ("mae", "pearson_r")  # , "spearman_rho", "r2_score", "mape", "mse")
    source_datasets = ("TDBRAIN",)
    hp = "DL architecture"
    line_styles = {"DL architecture": {"IN": (0, (5, 2, 10, 3)), "DN": ":", "SN": "-."},
                   "Band-pass filter": {"All": ":", "Delta": ":", "Theta": ":", "Alpha": ":", "Beta": ":",
                                        "Gamma": ":"}}
    colors = {"mae": "blue", "pearson_r": "orange", "spearman_rho": "orange", "r2_score": "green", "mape": "yellow",
              "mse": "orange"}
    alpha = 1

    results_dir = Path(get_results_dir())

    # Loop through all selection metrics
    for s, selection_metric in enumerate(selection_metrics):
        # ------------
        # Create dataframes with all information we want
        # ------------
        # Load the results df
        path = Path(os.path.dirname(__file__)) / f"all_test_results_selection_metric_{selection_metric}.csv"
        results_df = pandas.read_csv(path)

        # Extract subset
        if conditions:
            combined_conditions = combine_conditions(df=results_df, conditions=conditions)
            results_df = results_df[combined_conditions]

        # Add the configurations
        df = add_hp_configurations_to_dataframe(results_df, hps=(hp,), results_dir=results_dir)
        df.replace(renamed_df_mapping, inplace=True)

        # ------------
        # Get the data
        # ------------
        for source_dataset in source_datasets:
            # Create a subset
            source_subset_df = df[df["Source dataset"] == source_dataset]

            # Loop through all categories of the selected HP
            for h, hp_category in enumerate(set(source_subset_df[hp])):
                # Create subset
                subset_df = source_subset_df[source_subset_df[hp] == hp_category]

                # Create the y-axis
                x_axis = sorted(subset_df[target_metric].unique())
                cumulative = (numpy.array([sum(subset_df[target_metric] < threshold) for threshold in x_axis])
                              / subset_df.shape[0])

                # Plot
                selection_metric_label = selection_metric if h == 0 else None
                hp_category_label = hp_category if s == 0 else None

                pyplot.semilogx(x_axis, cumulative, label=selection_metric_label, color=colors[selection_metric],
                                linewidth=2)
                pyplot.semilogx(x_axis, cumulative,label=hp_category_label, color="black",
                                linestyle=line_styles[hp][hp_category], alpha=alpha)
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
