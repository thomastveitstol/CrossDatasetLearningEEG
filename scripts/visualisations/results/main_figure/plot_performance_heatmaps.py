import itertools
import os
from pathlib import Path
from typing import Literal, Union, Type

import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import combine_conditions, get_label_orders, higher_is_better, PRETTY_NAME


def main():
    selection_metrics = ("r2_score", "pearson_r", "mae")
    target_metrics = ("r2_score", "pearson_r", "mae")
    metric_combinations: Union[Type[zip], Type[itertools.product]] = zip
    conditions = {"Source dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang", "Pooled"),
                  "Target dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang", "Pooled")}
    value_ranges = {"r2_score": (-3, 1), "pearson_r": (-1, 1), "mae": (None, None)}


    orders = get_label_orders()
    formats = {"pearson_r": ".2f", "mae": ".1f", "r2_score": ".2f"}

    # ------------
    # Create dataframes with all information we want
    # ------------
    for selection_metric, target_metric in metric_combinations(selection_metrics, target_metrics):
        # Load the results df
        path = Path(os.path.dirname(__file__)) / f"all_test_results_selection_metric_{selection_metric}.csv"
        results_df = pandas.read_csv(path)

        # Extract subset
        if conditions:
            combined_conditions = combine_conditions(df=results_df, conditions=conditions)
            results_df = results_df[combined_conditions]

        # ------------
        # Get the best performing models only
        # ------------
        # Get dataset names
        _all_tar_datasets = set(results_df["Target dataset"])
        _all_sou_datasets = set(results_df["Source dataset"])

        target_datasets = tuple(d_name for d_name in orders["Target dataset"] if d_name in _all_tar_datasets)
        source_datasets = tuple(d_name for d_name in orders["Source dataset"] if d_name in _all_sou_datasets)

        test_scores = {"Target dataset": [], "Source dataset": [], "Test score": []}
        for source_dataset in source_datasets:
            # Loop through all target datasets
            for target_dataset in target_datasets:
                if target_dataset == source_dataset:
                    continue

                # Select the row based on validation performance
                subset_cond = ((results_df["Source dataset"] == source_dataset) &
                               (results_df["Target dataset"] == target_dataset))
                subset_results_df = results_df[subset_cond]
                if higher_is_better(selection_metric):
                    best_val_run = subset_results_df["run"].loc[subset_results_df["Val score"].idxmax()]
                else:
                    best_val_run = subset_results_df["run"].loc[subset_results_df["Val score"].idxmin()]

                # Compute source to target score
                _cond = (subset_cond & (results_df["run"] == best_val_run))
                score = results_df[target_metric].loc[_cond].iloc[0]

                # Add the results
                test_scores["Target dataset"].append(target_dataset)
                test_scores["Source dataset"].append(source_dataset)
                test_scores["Test score"].append(score)

        test_scores_df = pandas.DataFrame(test_scores)

        # test_scores_df.set_index("Source dataset", inplace=True)
        test_scores_df = test_scores_df.pivot(columns="Target dataset", index="Source dataset", values="Test score")

        # Reorder rows and columns
        t_order = tuple(dataset for dataset in orders["Target dataset"] if dataset in target_datasets)
        s_order = tuple(dataset for dataset in orders["Source dataset"] if dataset in source_datasets)

        test_scores_df = test_scores_df.reindex(index=s_order, columns=t_order)

        # Plotting
        pyplot.figure()
        seaborn.heatmap(
            test_scores_df, annot=True, cmap="viridis", cbar_kws={"label": PRETTY_NAME[target_metric]},
            fmt=formats[target_metric], vmin=value_ranges[target_metric][0], vmax=value_ranges[target_metric][1]
        )
        pyplot.title(f"Test scores (Selection metric: {PRETTY_NAME[selection_metric]})")

    pyplot.show()


if __name__ == "__main__":
    main()
