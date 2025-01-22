import os
from pathlib import Path

import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import add_hp_configurations_to_dataframe, PRETTY_NAME, combine_conditions, \
    get_label_orders, get_rename_mapping, extract_selected_best_scores, get_dummy_performance, INV_PRETTY_NAME
from cdl_eeg.data.paths import get_results_dir


def _fill_subplot(ax, *, col_figure_hp, color_hp, df, limits, orders, row_figure_hp, row_number, x_axis, x_fig_category,
                  y_axis, y_fig_category, add_y_category_to_label, selection_metric, dummy_performance):
    # Select the correct dataframe subset
    subset_df = df[(df[row_figure_hp] == y_fig_category) & (df[col_figure_hp] == x_fig_category)]

    # Fix orders
    hue_order = tuple(val for val in orders[color_hp] if val in set(subset_df[color_hp]))
    y_order = tuple(val for val in orders[y_axis] if val in set(subset_df[y_axis]))

    # -------------
    # Plotting
    # -------------
    # Box-plot and strip-plot
    seaborn.boxplot(
        data=subset_df, x=x_axis, y=y_axis, hue=color_hp, hue_order=hue_order, order=y_order, ax=ax,
        linewidth=1.2, dodge=True, showfliers=False, fill=False
    )
    seaborn.stripplot(
        data=subset_df, x=x_axis, y=y_axis, hue=color_hp, hue_order=hue_order, order=y_order, ax=ax, jitter=True,
        dodge=True, size=3, alpha=0.5, marker='o'
    )

    # Plot the selected model
    # Get the best score (selected by maximising performance on validation set)
    selected_df = extract_selected_best_scores(
        df=subset_df, selection_metric=selection_metric, target_metrics=(x_axis,),  # Assumes x-axis is target metric
        target_datasets=set(subset_df["Target dataset"]), source_datasets=set(subset_df["Source dataset"]),
        additional_columns=(color_hp, y_axis)
    )
    seaborn.stripplot(
        data=selected_df, x=f"Performance ({PRETTY_NAME[x_axis]})", y=y_axis, hue=color_hp, hue_order=hue_order,
        ax=ax, jitter=True, dodge=True, size=9, alpha=1, marker="*", palette="dark:black"
    )

    # Add a dummy performance line
    _target_datasets = set(subset_df["Target dataset"])
    _source_datasets = set(subset_df["Source dataset"])
    assert len(_target_datasets) == len(_source_datasets) == 1
    target_dataset = tuple(_target_datasets)[0]
    source_dataset = tuple(_source_datasets)[0]

    condition = ((dummy_performance["Target dataset"] == target_dataset)
                 & (dummy_performance["Source dataset"] == source_dataset))
    dummy_score = dummy_performance[f"Performance ({PRETTY_NAME[x_axis]})"][condition]
    ax.plot((dummy_score, dummy_score), (-0.5, len(y_order) - 0.5), color="black", linewidth=1.5, linestyle="--",
            alpha=0.8, zorder=5)

    # ------------
    # Plot cosmetics
    # ------------
    # Theme (shading with grey)
    for i, _ in enumerate(y_order):
        if i % 2 == 0:  # Shade alternate categories
            ax.axhspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.5)

    # Normal cosmetics
    ax.set_xlim(*limits[x_axis])
    try:
        ax.get_legend().remove()  # This may raise an AttributeError
    except AttributeError:  # This happens if plots are empty
        pass
    if row_number == 0:
        ax.set_title(x_fig_category)
    ax.grid()
    if add_y_category_to_label:
        ax.set_ylabel(f"{y_fig_category}\n{y_axis}")
    else:
        ax.set_ylabel(y_axis)
    if x_axis in PRETTY_NAME:
        ax.set_xlabel(PRETTY_NAME[x_axis])


def main():
    # ------------
    # Some choices
    # ------------
    experiment_type = "lodi"

    y_axis = "DL architecture"
    x_axis = "mae"
    color_hp = "Band-pass filter"  # The HP which will be used as 'hue'
    selection_metrics = (x_axis,)
    figsize = (11.5, 5)
    title_fontsize = 12

    renamed_df_mapping = get_rename_mapping()
    limits = {"pearson_r": (-0.2, 1), "mae": (0, 40), "r2_score": (-3, 1), "spearman_rho": (-0.2, 1)}
    orders = get_label_orders(renamed_df_mapping)

    # Conditions and columns and rows
    if experiment_type.lower() == "lodo":
        row_figure_hp = "Source dataset"  # One figure per category of this HP. Figures will be stacked i y direction
        col_figure_hp = "Target dataset"  # One figure per category of this HP. Figures will be stacked i x direction
        conditions = {"Source dataset": ("Pooled",)}
    elif experiment_type.lower() == "lodi":
        row_figure_hp = "Target dataset"  # One figure per category of this HP. Figures will be stacked i y direction
        col_figure_hp = "Source dataset"  # One figure per category of this HP. Figures will be stacked i x direction
        conditions = {"Target dataset": ("Pooled",)}
    else:
        raise ValueError(f"Unexpected experiment type: {experiment_type}")

    results_dir = Path(get_results_dir())
    _datasets = tuple(INV_PRETTY_NAME[name] for name in ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang"))
    dummy_performance = get_dummy_performance(datasets=_datasets, metrics=(x_axis,))

    for selection_metric in selection_metrics:
        # ------------
        # Create dataframes with all information we want
        # ------------
        # Load the results df
        path = Path(os.path.dirname(__file__)) / f"all_test_results_selection_metric_{selection_metric}.csv"
        results_df = pandas.read_csv(path)

        # Extract subset
        if conditions:
            _current_conditions = {col: values for col, values in conditions.items() if col in results_df.columns}
            combined_conditions = combine_conditions(df=results_df, conditions=_current_conditions)
            results_df = results_df[combined_conditions]

        # Add the configurations
        df = add_hp_configurations_to_dataframe(results_df, hps={color_hp, row_figure_hp, y_axis, *conditions.keys()},
                                                results_dir=results_dir)
        df.replace(renamed_df_mapping, inplace=True)

        # Extract subset again
        if conditions:
            combined_conditions = combine_conditions(df=df, conditions=conditions)
            df = df[combined_conditions]

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
        fig, axes = pyplot.subplots(len(y_fig_categories), len(x_fig_categories), figsize=figsize,
                                    sharex=True, sharey=True)  # type: ignore
        axes = (axes,) if axes.ndim == 1 else axes  # Fix the spe

        # Loop through the axes to create subplots
        for row_number, (axes_row, y_fig_category) in enumerate(zip(axes, y_fig_categories)):
            # Loop through the columns
            for ax, x_fig_category in zip(axes_row, x_fig_categories):
                _fill_subplot(
                    ax, col_figure_hp=col_figure_hp, color_hp=color_hp, df=df, limits=limits, orders=orders,
                    row_figure_hp=row_figure_hp, row_number=row_number, x_axis=x_axis,
                    x_fig_category=x_fig_category, y_axis=y_axis, y_fig_category=y_fig_category,
                    add_y_category_to_label=len(y_fig_categories) > 1, selection_metric=selection_metric,
                    dummy_performance=dummy_performance
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
