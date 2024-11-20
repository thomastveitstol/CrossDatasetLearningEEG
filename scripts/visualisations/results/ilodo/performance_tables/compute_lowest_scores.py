"""
In the abstract of the paper, I will need the lowest LODI score. However, I've also decided (at least per now) to use
the selection metric which gave the highest performance (and write that the performance scores were 'up to').

To avoid human errors I made a script for this
"""
import os

import numpy
import pandas

from cdl_eeg.data.analysis.results_analysis import PRETTY_NAME


def _get_dataframes(metric):
    # Path
    root_dir = os.path.dirname(__file__)

    # Read all csv files
    df_mae = pandas.read_csv(os.path.join(root_dir, "results_mae.csv"))
    df_mse = pandas.read_csv(os.path.join(root_dir, "results_mse.csv"))
    df_pearson = pandas.read_csv(os.path.join(root_dir, "results_pearson_r.csv"))
    df_r2 = pandas.read_csv(os.path.join(root_dir, "results_r2_score.csv"))

    # Remove all performance scores which is not the current metric of interest (Pearson's r and R2 was used in the
    # paper)
    df_mae = df_mae[df_mae["metric"] == metric]
    df_mse = df_mse[df_mse["metric"] == metric]
    df_pearson = df_pearson[df_pearson["metric"] == metric]
    df_r2 = df_r2[df_r2["metric"] == metric]

    # Remove the metric column
    df_mae.drop("metric", axis="columns", inplace=True)
    df_mse.drop("metric", axis="columns", inplace=True)
    df_pearson.drop("metric", axis="columns", inplace=True)
    df_r2.drop("metric", axis="columns", inplace=True)

    # Set source dataset to index
    df_mae.set_index("source_dataset", inplace=True)
    df_mse.set_index("source_dataset", inplace=True)
    df_pearson.set_index("source_dataset", inplace=True)
    df_r2.set_index("source_dataset", inplace=True)

    return {"mae": df_mae, "mse": df_mse, "pearson": df_pearson, "r2": df_r2}

def main():

    metrics_of_interest = ("pearson_r", "r2_score")
    datasets = ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang")

    # Initialise the best values (just using Pearson and R2, so higher is better)
    maximums = {metric: {f"{source_dataset} -> {target_dataset}": None for source_dataset in datasets
                         for target_dataset in datasets} for metric in metrics_of_interest}

    # Loop through metrics of interest
    for metric in metrics_of_interest:
        for selection_metric, df in _get_dataframes(metric=metric).items():
            for source_dataset in datasets:
                for target_dataset in datasets:
                    score = df.loc[source_dataset, target_dataset]
                    best_score = maximums[metric][f"{source_dataset} -> {target_dataset}"]
                    if best_score is None or score > best_score:  # Higher is better for the metrics in use
                        maximums[metric][f"{source_dataset} -> {target_dataset}"] = score

    # Print in a prettier way
    for metric, scores in maximums.items():
        print(f"\n\n{f' Metric: {PRETTY_NAME[metric]} ':=^30}")
        for dataset_combination, performance in scores.items():
            print(f"{dataset_combination}: {performance}")

        print(f"\nSmallest maximum ({PRETTY_NAME[metric]}): "
              f"{min(tuple(s for s in scores.values() if not numpy.isnan(s)))}")  # type: ignore

if __name__ == "__main__":
    main()
