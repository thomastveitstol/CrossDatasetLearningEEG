import os.path
from pathlib import Path

import pandas
import seaborn
from matplotlib import pyplot


def main():
    # ------------
    # Some choices
    # ------------
    metrics = ("pearson_r", "mae", "r2_score")
    limits = {"pearson_r": (-0.2, 1), "mae": (0, 70), "r2_score": (-4, 1), "spearman_rho": (-0.2, 1)}
    dataset_order = ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang", "Pooled")

    # ------------
    # Plotting
    # ------------
    # Load df
    path = Path(os.path.dirname(__file__)) / "all_test_results_selection_metric_pearson_r.csv"
    df = pandas.read_csv(path)

    # Get all dataset names (needed for legends)
    dataset_names = set(df["Source dataset"])

    # Create the plot with transposed layout
    fig, axes = pyplot.subplots(1, 3, figsize=(10, 12), sharey=True)

    # Loop through the axes to create subplots
    for ax, metric in zip(axes.flat, metrics):
        seaborn.boxplot(
            x=metric, y="Target dataset", hue="Source dataset", data=df, hue_order=dataset_order, order=dataset_order,
            ax=ax, linewidth=1.2, dodge=True, showfliers=False, fill=False #boxprops=dict(facecolor='none')
        )
        seaborn.stripplot(
            x=metric, y="Target dataset", hue="Source dataset", data=df, hue_order=dataset_order, order=dataset_order,
            ax=ax, jitter=True, dodge=True, size=3, alpha=0.5, marker='o'
        )
        ax.set_xlim(*limits[metric])
        ax.get_legend().remove()
        ax.grid()

    # Customizing legends separately
    # Add a single legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles[:len(dataset_names)], labels[:len(dataset_names)],
        loc='upper center', ncol=len(dataset_names),
        bbox_to_anchor=(0.5, 0.97), frameon=False
    )
    #fig.legend(loc='outside upper right')
    """fig.legend(
        labels=dataset_names, loc='upper center', ncol=len(dataset_names),
        bbox_to_anchor=(0.5, 0.97), frameon=False
    )"""

    # Layout and display
    pyplot.tight_layout()
    pyplot.subplots_adjust(top=0.9)  # Adjust space for the global legend
    pyplot.show()


if __name__ == "__main__":
    main()
