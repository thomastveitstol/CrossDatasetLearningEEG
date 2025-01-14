import os
from pathlib import Path

import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import add_hp_configurations_to_dataframe
from cdl_eeg.data.paths import get_results_dir


def main():
    # ------------
    # Some choices
    # ------------
    y_axis = "Source dataset"
    x_axis = "pearson_r"
    color_hp = "Band-pass filter"  # The HP which will be used as 'hue'
    row_figure_hp = "DL architecture"  # One figure per category of this HP. Figures will be stacked i y direction
    col_figure_hp = "Target dataset"  # One figure per category of this HP. Figures will be stacked i x direction

    title_fontsize = 14

    limits = {"pearson_r": (-0.2, 1), "mae": (0, 70), "r2_score": (-4, 1), "spearman_rho": (-0.2, 1)}
    orders = {"Target dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang", "Pooled"),
              "Source dataset": ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang", "Pooled"),
              "Band-pass filter": ("All", "Delta", "Theta", "Alpha", "Beta", "Gamma"),
              "DL architecture": ("InceptionNetwork", "Deep4NetMTS", "ShallowFBCSPNetMTS")}

    results_dir = Path(get_results_dir())

    # ------------
    # Create dataframes with all information we want
    # ------------
    # Load the results df
    path = Path(os.path.dirname(__file__)) / "all_test_results_selection_metric_pearson_r.csv"
    results_df = pandas.read_csv(path)

    # Add the configurations
    df = add_hp_configurations_to_dataframe(results_df, hps=(color_hp, row_figure_hp), results_dir=results_dir)

    # Get all dataset names
    y_fig_categories = orders[row_figure_hp] if row_figure_hp in orders else set(df[row_figure_hp])
    x_fig_categories = orders[col_figure_hp] if col_figure_hp in orders else set(df[col_figure_hp])

    # ------------
    # Plotting
    # ------------
    # Create the plot
    fig, axes = pyplot.subplots(len(y_fig_categories), len(x_fig_categories), figsize=(10, 12), sharex=True,
                                sharey=True)

    # Loop through the axes to create subplots
    for row_number, (axes_row, y_fig_category) in enumerate(zip(axes, y_fig_categories)):
        # Loop through the columns
        for ax, x_fig_category in zip(axes_row, x_fig_categories):
            # Select the correct dataframe subset
            subset_df = df[(df[row_figure_hp] == y_fig_category) & (df[col_figure_hp] == x_fig_category)]

            # Plotting
            seaborn.boxplot(
                x=subset_df[x_axis], y=y_axis, hue=color_hp, hue_order=orders[color_hp], order=orders[y_axis],
                data=subset_df, ax=ax, linewidth=1.2, dodge=True, showfliers=False, fill=False
            )
            seaborn.stripplot(
                x=subset_df[x_axis], y=y_axis, hue=color_hp, hue_order=orders[color_hp], order=orders[y_axis],
                data=subset_df, ax=ax, jitter=True, dodge=True, size=3, alpha=0.5, marker='o'
            )
            ax.set_xlim(*limits[x_axis])
            ax.get_legend().remove()
            if row_number == 0:
                ax.set_title(x_fig_category)
            ax.grid()
            ax.set_ylabel(f"{y_fig_category}\n\n{y_axis}")

    fig.suptitle(col_figure_hp, fontsize=title_fontsize)

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    _legend_names = set(df[color_hp])
    fig.legend(
        handles[:len(_legend_names)], labels[:len(_legend_names)],
        loc="lower center", ncol=len(_legend_names),
        bbox_to_anchor=(0.5, 0.03), frameon=False
    )

    # Layout and display
    pyplot.tight_layout()
    pyplot.subplots_adjust(bottom=0.1)  # Adjust space for the global legend
    pyplot.show()


if __name__ == "__main__":
    main()
