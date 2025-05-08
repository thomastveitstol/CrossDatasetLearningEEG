import os.path
from pathlib import Path

import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import combine_conditions, get_label_orders, PRETTY_NAME


def main():
    metric = "pearson_r"

    for experiment_type in ("lodo", "lodi"):
        figsize = (7, 5)
        fontsize = 16
        title_fontsize = fontsize + 5

        # More fixing
        if experiment_type.lower() == "lodo":
            x_lims = {"pearson_r": (-0.28, 1)}
            y_lims = {"pearson_r": (-0.6, 1)}
            row_dataset = "Target dataset"
            hue = None
            hue_order = None
            colors = None
            conditions = {"Source dataset": ("Pooled",)}
        elif experiment_type.lower() == "lodi":
            x_lims = {"pearson_r": (-0.75, 1)}
            y_lims = {"pearson_r": (-0.6, 1)}
            row_dataset = "Source dataset"
            hue_order = ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang")
            hue = "Target dataset"
            colors = dict(zip(hue_order, seaborn.color_palette("tab10", len(hue_order))))
            conditions ={"Source dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang")}
        else:
            raise ValueError(f"Unexpected experiment type: {experiment_type}")

        # -----------
        # Load data
        # -----------
        path = Path(os.path.dirname(os.path.dirname(__file__))) / f"all_test_results_selection_metric_{metric}.csv"
        df = pandas.read_csv(path)

        # Extract subset
        if conditions:
            combined_conditions = combine_conditions(df=df, conditions=conditions)
            df = df[combined_conditions]

        all_datasets = set(df[row_dataset])
        dataset_order = (dataset for dataset in get_label_orders()["Target dataset"] if dataset in all_datasets)
        for dataset in dataset_order:
            pyplot.figure(figsize=figsize)
            subset_df = df[df[row_dataset] == dataset]

            if hue_order is not None:
                curr_hue_order = (level for level in hue_order if level != dataset)
            else:
                curr_hue_order = None
            seaborn.scatterplot(data=subset_df, x="Val score", y=metric, hue=hue, hue_order=curr_hue_order, palette=colors)

            # Cosmetics
            pyplot.title(f"{row_dataset}: {dataset}", fontsize=title_fontsize)
            pyplot.ylabel(f"Test performance ({PRETTY_NAME[metric]})", fontsize=fontsize)
            pyplot.xlabel(f"Validation performance ({PRETTY_NAME[metric]})", fontsize=fontsize)
            pyplot.tick_params(labelsize=fontsize)
            pyplot.xlim(x_lims[metric])
            pyplot.ylim(y_lims[metric])
            pyplot.grid()
            if hue:
                pyplot.legend(fontsize=fontsize)
            pyplot.tight_layout()

            file_name = f"{experiment_type}_{dataset.lower()}.png"
            pyplot.savefig(Path(os.path.dirname(__file__)) / file_name)


if __name__ == "__main__":
    main()