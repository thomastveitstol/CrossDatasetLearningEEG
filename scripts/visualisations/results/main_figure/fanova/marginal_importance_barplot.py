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

    # To percentage
    df.loc[:, ~df.columns.isin(("Target dataset", "Source dataset", "Percentile", "HP"))] *= 100

    # Select percentile
    if percentile not in df["Percentile"]:
        raise ValueError(f"The HP importance for the selected percentile ({percentile}) has not been computed. Either "
                         f"use an available percentile {set(df['Percentile'])} or re-run the fANOVA analysis")
    df = df[df["Percentile"] == percentile]

    return df


def _get_pairwise_importance_df(experiment_type, *, percentile, selection_metric, target_metric, hp_pairs, value_name):
    file_name = f"pairwise_importance_{experiment_type}_{selection_metric}_{target_metric}.csv"
    path = Path(os.path.dirname(__file__)) / file_name
    df = pandas.read_csv(path, index_col=0)

    df.rename(columns={"Importance": value_name}, inplace=True)

    # For LODI, just using source dataset to pooled target
    if experiment_type.lower() == "lodi":
        df = df[df["Target dataset"] == "Pooled"]

    # To percentage
    df.loc[:, ~df.columns.isin(("Target dataset", "Source dataset", "Percentile", "Rank", "HP1", "HP2"))] *= 100

    # Select percentile
    if percentile not in df["Percentile"]:
        raise ValueError(f"The HP importance for the selected percentile ({percentile}) has not been computed. Either "
                         f"use an available percentile {set(df['Percentile'])} or re-run the fANOVA analysis")
    df = df[df["Percentile"] == percentile]

    # Select pair
    sorted_hp_pairs = tuple(tuple(sorted(hp_pair)) for hp_pair in hp_pairs)
    df = pandas.concat((df[(df["HP1"] == hp_1) & (df["HP2"] == hp_2)] for hp_1, hp_2 in sorted_hp_pairs), axis="rows")

    df["HP"] = df["HP1"] + "\n+ " + df["HP2"]
    df.drop(["HP1", "HP2"], inplace=True, axis="columns")
    return df


def main():
    # -------------
    # Some selections
    # -------------
    experiment_types = ("lodo", "lodi")
    selection_metric = "mae"
    target_metric = selection_metric
    percentiles = (0, 50, 75, 90, 95)

    y_lim = (0, None)
    fontsize = 12
    title_fontsize = fontsize + 3
    figsize = (14, 6.25)
    palette = "tab20"
    linewidth = 1
    value_name = "Importance (%)"
    save_path = Path(os.path.dirname(__file__)) / "marginal_importance_plots"
    hp_pairs = (("DL architecture", "Band-pass filter"), ("Band-pass filter", "Normalisation"),
                ("Band-pass filter", "Spatial method"))

    for percentile in percentiles:
        # -------------
        # Load data
        # -------------
        dataframes = {}
        for experiment_type in experiment_types:
            # Get the marginal effects
            marginal_df = _get_marginal_importance_df(
                experiment_type, percentile=percentile, selection_metric=selection_metric, target_metric=target_metric,
                value_name=value_name
            )
            if not hp_pairs:
                df = marginal_df
            else:
                # Get the pairwise effects
                pairwise_df = _get_pairwise_importance_df(
                    experiment_type, percentile=percentile, selection_metric=selection_metric, target_metric=target_metric,
                    hp_pairs=hp_pairs, value_name=value_name
                )
                pairwise_df.drop("Rank", inplace=True, axis="columns")

                df = pandas.concat((marginal_df, pairwise_df), axis="rows")

            dataframes[experiment_type] = df.melt(
                id_vars=("Target dataset", "Source dataset", "Percentile", "Mean", "Std", "HP"), var_name="Tree",
                value_name=value_name
            )
            print(dataframes[experiment_type])



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
        # noinspection PyTypeChecker
        fig, axes = pyplot.subplots(1, len(experiment_types), figsize=figsize, sharey=True)
        for ax, (experiment_type, df) in zip(axes, dataframes.items()):
            if experiment_type.lower() == "lodo":
                x_axis = "Target dataset"
            elif experiment_type.lower() == "lodi":
                x_axis = "Source dataset"
            else:
                raise ValueError(f"Unexpected experiment type: {experiment_type}")

            # Plot
            _all_x_levels = set(df[x_axis])
            x_axis_order = tuple(x for x in get_label_orders()[x_axis] if x in _all_x_levels)
            seaborn.boxplot(df, x=x_axis, y=value_name, hue="HP", order=x_axis_order, linewidth=linewidth, ax=ax,
                            palette=palette_mapping, showfliers=False)
            ax.get_legend().remove()

            # Cosmetics
            ax.set_title(experiment_type.upper(), weight="bold", fontsize=title_fontsize)
            ax.grid()
            ax.set_xlim(-0.5, len(set(df[x_axis])) - 0.5)
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
