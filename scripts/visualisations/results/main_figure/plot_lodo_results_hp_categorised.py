import os
from pathlib import Path

import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import add_hp_configurations_to_dataframe, PRETTY_NAME, combine_conditions, \
    get_label_orders, get_rename_mapping
from cdl_eeg.data.paths import get_results_dir


def _fill_subplot(ax, *, col_figure_hp, color_hp, df, limits, orders, row_figure_hp, row_number, x_axis, x_fig_category,
                  y_axis, y_fig_category, add_y_category_to_label):
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
    try:
        ax.get_legend().remove()  # This may raise an AttributeError
    except AttributeError:
        # This happens if plots are empty
        pass
    if row_number == 0:
        ax.set_title(x_fig_category)
    ax.grid()
    if add_y_category_to_label:
        ax.set_ylabel(f"{y_fig_category}\n{y_axis}")
    else:
        ax.set_ylabel(y_axis)


def main():
    # ------------
    # Some choices
    # ------------
    y_axis = "DL architecture"
    x_axis = "mae"
    color_hp = "Band-pass filter"  # The HP which will be used as 'hue'
    row_figure_hp = "Target dataset"  # One figure per category of this HP. Figures will be stacked i y direction
    col_figure_hp = "Source dataset"  # One figure per category of this HP. Figures will be stacked i x direction
    selection_metrics = ("mae", "pearson_r")
    figsize = (10, 5)
    conditions = {"Target dataset": ("Pooled",)}
    renamed_df_mapping = get_rename_mapping()

    title_fontsize = 14

    limits = {"pearson_r": (-0.2, 1), "mae": (0, 40), "r2_score": (-3, 1), "spearman_rho": (-0.2, 1)}
    orders = get_label_orders()
    # Update 'orders' with the 'renamed_df_mapping'
    for column, mapping in renamed_df_mapping.items():
        if column in orders:
            updated_order = []
            for value in orders[column]:
                if value in mapping:
                    updated_order.append(mapping[value])
                else:
                    updated_order.append(value)
            orders[column] = tuple(updated_order)

    results_dir = Path(get_results_dir())

    for selection_metric in selection_metrics:
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
        df = add_hp_configurations_to_dataframe(results_df, hps=(color_hp, row_figure_hp, y_axis),
                                                results_dir=results_dir)
        df.replace(renamed_df_mapping, inplace=True)

        # Get all dataset names
        y_fig_categories = orders[row_figure_hp] if row_figure_hp in orders else set(df[row_figure_hp])
        x_fig_categories = orders[col_figure_hp] if col_figure_hp in orders else set(df[col_figure_hp])

        # Remove excessive which may have been removed by conditions
        y_fig_categories = tuple(cat for cat in y_fig_categories if cat in set(df[row_figure_hp]))
        x_fig_categories = tuple(cat for cat in x_fig_categories if cat in set(df[col_figure_hp]))

        # ------------
        # Plotting
        # ------------
        # Create the plot
        fig, axes = pyplot.subplots(len(y_fig_categories), len(x_fig_categories), figsize=figsize, sharex=True,
                                    sharey=True)
        axes = (axes,) if axes.ndim == 1 else axes  # Fix the spe

        # Loop through the axes to create subplots
        for row_number, (axes_row, y_fig_category) in enumerate(zip(axes, y_fig_categories)):
            # Loop through the columns
            for ax, x_fig_category in zip(axes_row, x_fig_categories):
                _fill_subplot(
                    ax, col_figure_hp=col_figure_hp, color_hp=color_hp, df=df, limits=limits, orders=orders,
                    row_figure_hp=row_figure_hp, row_number=row_number, x_axis=x_axis,
                    x_fig_category=x_fig_category, y_axis=y_axis, y_fig_category=y_fig_category,
                    add_y_category_to_label=len(y_fig_categories) > 1
                )


        fig.suptitle(f"{col_figure_hp} (Selection metric: {PRETTY_NAME[selection_metric]})", fontsize=title_fontsize)
        if len(y_fig_categories) > 1:
            fig.supylabel(row_figure_hp)

        handles, labels = axes[0][0].get_legend_handles_labels()
        _legend_names = set(df[color_hp])
        fig.legend(
            handles[:len(_legend_names)], labels[:len(_legend_names)],
            loc="right", # ncol=len(_legend_names),
            bbox_to_anchor=(1.01, 0.5), frameon=False
        )

        # Layout and display
        fig.tight_layout()
        fig.subplots_adjust(right=0.9)  # Adjust space for the global legend
    pyplot.show()


if __name__ == "__main__":
    main()
