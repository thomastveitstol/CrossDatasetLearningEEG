import os
from pathlib import Path

import matplotlib
import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import get_label_orders


def _get_marginal_importance_df(experiment_type, *, percentile, selection_metric, target_metric, value_name):
    file_name = f"marginal_importance_{experiment_type}_{selection_metric}_{target_metric}.csv"
    path = Path(os.path.dirname(__file__)) / file_name
    df = pandas.read_csv(path, index_col=0)

    # Not using the standard deviations for now
    df.drop(labels=[col_name for col_name in df.columns if col_name.endswith("(std)")], axis="columns", inplace=True)

    # For LODI, just using source dataset to pooled target
    if experiment_type.lower() == "lodi":
        df = df[df["Target dataset"] == "Pooled"]

    # Also, remove '(mean)'
    _suffix = " (mean)"
    df.rename(columns={col_name: col_name[:-len(_suffix)] for col_name in df.columns if col_name.endswith(_suffix)},
              inplace=True)

    # To long format
    df = df.melt(id_vars=("Target dataset", "Source dataset", "Percentile"), var_name="HP", value_name=value_name)

    # To percentage
    df[value_name] *= 100

    # Select percentile
    if percentile not in df["Percentile"]:
        raise ValueError(f"The HP importance for the selected percentile ({percentile}) has not been computed. Either "
                         f"use an available percentile {set(df['Percentile'])} or re-run the fANOVA analysis")
    df = df[df["Percentile"] == percentile]

    return df


def main():
    # -------------
    # Some selections
    # -------------
    experiment_types = ("lodo", "lodi")
    selection_metric = "pearson_r"
    target_metric = selection_metric
    percentiles = (0, 50, 75, 90, 95)

    y_lim = (0, 18)
    fontsize = 12
    title_fontsize = fontsize + 3
    figsize = (14, 7)
    palette = "Spectral"
    linewidth = 1
    value_name = "Importance (%)"
    save_path = Path(os.path.dirname(__file__)) / "marginal_importance_plots"

    for percentile in percentiles:
        # -------------
        # Load data
        # -------------
        dataframes = {}
        for experiment_type in experiment_types:
            dataframes[experiment_type] = _get_marginal_importance_df(
                experiment_type, percentile=percentile, selection_metric=selection_metric, target_metric=target_metric,
                value_name=value_name
            )

        # Use the color palette from LODO because it has more HPs (tau)
        unique_hps = []  # Use sorting as in the df
        for hp_name in dataframes["lodo"]["HP"]:
            if hp_name not in unique_hps:
                unique_hps.append(hp_name)
        color_palette = seaborn.color_palette(palette=palette, n_colors=len(unique_hps))
        palette_mapping = {hp_name: color for color, hp_name in zip(color_palette, unique_hps)}

        # -------------
        # Plotting
        # -------------
        matplotlib.rcParams.update({'font.size': fontsize})
        fig, axes = pyplot.subplots(len(experiment_types), 1, figsize=figsize)
        for ax, (experiment_type, df) in zip(axes, dataframes.items()):
            if experiment_type.lower() == "lodo":
                x_axis = "Target dataset"
            elif experiment_type.lower() == "lodi":
                x_axis = "Source dataset"
            else:
                raise ValueError(f"Unexpected experiment type: {experiment_type}")

            # Plot
            seaborn.barplot(df, x=x_axis, y=value_name, hue="HP", order=get_label_orders()[x_axis], linewidth=linewidth,
                            edgecolor="black", ax=ax, palette=palette_mapping)
            ax.get_legend().remove()

            # Cosmetics
            ax.set_title(experiment_type.upper(), weight="bold", fontsize=title_fontsize)
            ax.grid()
            ax.set_ylim(*y_lim)
            ax.tick_params(labelsize=fontsize)

            # Theme (shading with grey)
            for i, _ in enumerate(set(df[x_axis])):
                if i % 2 == 0:  # Shade alternate categories
                    ax.axvspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.5, zorder=-1)

        # Add legend
        fig.tight_layout()
        fig.subplots_adjust(right=0.827)  # Adjust space for the global legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels,loc="right", bbox_to_anchor=(1.001, 0.5), frameon=True, title="HP")

        # Save figure
        fig_name = f"fanova_marginal_hp_importance_percentile_{percentile}.png"
        fig.savefig(save_path / fig_name)


if __name__ == "__main__":
    main()
