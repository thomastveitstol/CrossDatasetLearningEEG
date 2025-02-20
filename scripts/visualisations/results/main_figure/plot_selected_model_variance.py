import itertools
import os
from pathlib import Path

import pandas
import seaborn
from matplotlib import pyplot, patches, lines

from cdl_eeg.data.analysis.results_analysis import combine_conditions, get_label_orders, extract_selected_best_scores, \
    add_hp_configurations_to_dataframe, get_rename_mapping, PRETTY_NAME
from cdl_eeg.data.paths import get_results_dir


def _get_scores_df(color_hp, conditions, orders, renamed_df_mapping, results_dir, selection_metrics, target_metrics):
    scores = {"Target dataset": [], "Source dataset": [], "Selection metric": [], color_hp: [], "Run": [],
              **{f"Performance ({PRETTY_NAME[metric]})": [] for metric in target_metrics},
              **{f"Std ({PRETTY_NAME[metric]})": [] for metric in target_metrics}}
    for selection_metric in selection_metrics:
        # Load the results df
        path = Path(os.path.dirname(__file__)) / f"all_test_results_selection_metric_{selection_metric}.csv"
        df = pandas.read_csv(path)

        # Extract subset
        if conditions:
            _current_conditions = {col: values for col, values in conditions.items() if col in df.columns}
            combined_conditions = combine_conditions(df=df, conditions=_current_conditions)
            df = df[combined_conditions]

        # Add HP(s)
        df = add_hp_configurations_to_dataframe(df=df, hps=(color_hp,), results_dir=results_dir)
        df.replace(renamed_df_mapping, inplace=True)

        # Extract subset again
        if conditions:
            combined_conditions = combine_conditions(df=df, conditions=conditions)
            df = df[combined_conditions]

        # Get dataset names
        _all_tar_datasets = set(df["Target dataset"])
        _all_sou_datasets = set(df["Source dataset"])

        target_datasets = tuple(d_name for d_name in orders["Target dataset"] if d_name in _all_tar_datasets)
        source_datasets = tuple(d_name for d_name in orders["Source dataset"] if d_name in _all_sou_datasets)

        # ------------
        # Get the best performing models only
        # ------------
        categories = orders[color_hp]
        for category in categories:
            # Extract subset
            subset_df = df[df[color_hp] == category]

            # Get unbiased test scores and standard deviations
            test_scores = extract_selected_best_scores(
                df=subset_df, selection_metric=selection_metric, target_metrics=target_metrics,
                target_datasets=target_datasets, source_datasets=source_datasets, include_std=True
            )
            #print(pandas.DataFrame(test_scores))

            # Add results
            lengths = set()
            for column, values in test_scores.items():
                scores[column].extend(values)
                lengths.add(len(values))
            assert len(lengths) == 1, len(lengths)
            num_rows = tuple(lengths)[0]

            scores["Selection metric"].extend([selection_metric] * num_rows)
            scores[color_hp].extend([category] * num_rows)

    return pandas.DataFrame(scores)


def main():
    selection_metrics = ("r2_score", "pearson_r", "mae", "mse")
    target_metrics = ("mae", "pearson_r")
    color_hp = "DL architecture"
    y_axis = "Target dataset"
    conditions = {"Source dataset": ("Pooled",)}

    stripedgecolor = "black"
    striplinewidth = 1.0
    size = 5
    jitter = True
    alpha = 0.7
    box_alpha = 0.7
    markers = {"r2_score": "s", "pearson_r": "^", "mae": "o", "mse": "*"}
    limits = {"pearson_r": (-0.2, 1), "mae": (0, 40), "r2_score": (-3, 1), "spearman_rho": (-0.2, 1)}

    renamed_df_mapping = get_rename_mapping()
    orders = get_label_orders(renamed_df_mapping)
    results_dir = Path(get_results_dir())
    strippalette = {category: color for category, color in zip(orders[color_hp], seaborn.color_palette())}

    # ------------
    # Get the dataframe with unbiased estimates and variances
    # ------------
    scores_df = _get_scores_df(
        color_hp=color_hp, conditions=conditions, orders=orders,  renamed_df_mapping=renamed_df_mapping,
        results_dir=results_dir, selection_metrics=selection_metrics, target_metrics=target_metrics
    )
    # con = (scores_df["DL architecture"] == "IN") & (scores_df["Source dataset"] == "LEMON")
    # print(scores_df[["Performance (MAE)", "Run"]][con])

    # ------------
    # Plotting
    # ------------
    # Fix orders
    hue_order = tuple(val for val in orders[color_hp] if val in set(scores_df[color_hp]))
    y_order = tuple(val for val in orders[y_axis] if val in set(scores_df[y_axis]))

    fig, axes = pyplot.subplots(nrows=len(target_metrics), ncols=2)
    for target_metric, row_axes in zip(target_metrics, axes):
        for x_axis, ax in zip((f"Std ({PRETTY_NAME[target_metric]})", f"Performance ({PRETTY_NAME[target_metric]})"),
                              row_axes):
            for selection_metric in selection_metrics:
                seaborn.stripplot(
                    x=x_axis, y=y_axis, hue=color_hp, hue_order=hue_order, order=y_order,
                    data=scores_df[scores_df["Selection metric"] == selection_metric], jitter=jitter, dodge=True,
                    size=size, alpha=alpha, marker=markers[selection_metric], edgecolor=stripedgecolor,
                    linewidth=striplinewidth, palette=strippalette, legend=False, ax=ax
                )
            # Add a boxplot
            seaborn.boxplot(
                x=x_axis, y=y_axis, hue=color_hp, hue_order=hue_order, order=y_order, data=scores_df, dodge=True,
                boxprops={"alpha": box_alpha}, showfliers=False, legend=False, palette=strippalette, ax=ax
            )

            # ------------
            # Plot cosmetics
            # ------------
            # Theme (shading with grey)
            for i, _ in enumerate(y_order):
                if i % 2 == 0:  # Shade alternate categories
                    ax.axhspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.3)

            # Normal cosmetics
            ax.grid(axis="x")
            ax.set_xlim(*limits[target_metric])

    # Fixing legends
    marker_handles = tuple(
        lines.Line2D([0], [0], marker=markers[selection_metric], color="black", label=PRETTY_NAME[selection_metric],
                     linestyle="None")
        for selection_metric in selection_metrics
    )
    color_handles = tuple(patches.Patch(color=strippalette[category], label=category, alpha=box_alpha)
                          for category in orders[color_hp])

    marker_legend = pyplot.legend(handles=marker_handles, title="Selection metric", loc="best")
    pyplot.gca().add_artist(marker_legend)
    pyplot.legend(handles=color_handles, title=color_hp, loc="best")

    pyplot.show()


if __name__ == "__main__":
    main()
