import dataclasses
import os
from pathlib import Path

import matplotlib
import pandas
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import combine_conditions, get_label_orders, PRETTY_NAME, \
    get_formats, extract_selected_best_scores
from cdl_eeg.data.analysis.results_analysis import get_dummy_performance, INV_PRETTY_NAME, higher_is_better


@dataclasses.dataclass(frozen=True)
class _Metric:
    selection_metric: str  # Metric used for model selection
    target_metric: str  # Metric used for measuring test performance
    subtract_dummy_performance: bool  # If dummy performance should be subtracted from the performance score


def main():
    metrics = (
        _Metric(selection_metric="mae", target_metric="mae", subtract_dummy_performance=True),
        _Metric(selection_metric="r2_score", target_metric="r2_score", subtract_dummy_performance=False),
        _Metric(selection_metric="r2_score", target_metric="r2_score_refit", subtract_dummy_performance=False),
        _Metric(selection_metric="pearson_r", target_metric="pearson_r", subtract_dummy_performance=False)
    )

    conditions = {}
    value_ranges = {"r2_score": (-3, 1), "pearson_r": (-1, 1), "mae": (None, None), "r2_score_refit": (-3, 1)}
    plot_kwargs = {
        "r2_score": {"cmap": "coolwarm", "norm": matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-3, vmax=1)},
        "pearson_r": {"cmap": "coolwarm", "center": 0},
        "mae": {"cmap": "coolwarm", "center": 0},
        "r2_score_refit": {"cmap": "coolwarm", "norm": matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-3, vmax=1)}}

    orders = get_label_orders()
    formats = get_formats()
    save_path = Path(os.path.dirname(__file__))

    # Maybe load the dummy performance
    subtract_metrics = tuple(metric.target_metric for metric in metrics if metric.subtract_dummy_performance)
    if subtract_metrics:
        _datasets = tuple(INV_PRETTY_NAME[d_name] for d_name in ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang"))
        dummy_df = get_dummy_performance(datasets=_datasets, metrics=subtract_metrics)
    else:
        dummy_df = None

    # ------------
    # Create dataframes with all information we want
    # ------------
    for metric in metrics:
        selection_metric = metric.selection_metric
        target_metric = metric.target_metric

        # Load the results df
        _root_path = os.path.dirname(os.path.dirname(__file__))
        path = Path(_root_path) / f"all_test_results_selection_metric_{selection_metric}.csv"
        results_df = pandas.read_csv(path)

        # Extract subset
        if conditions:
            combined_conditions = combine_conditions(df=results_df, conditions=conditions)
            results_df = results_df[combined_conditions]

        # Get dataset names
        _all_tar_datasets = set(results_df["Target dataset"])
        _all_sou_datasets = set(results_df["Source dataset"])

        target_datasets = tuple(d_name for d_name in orders["Target dataset"] if d_name in _all_tar_datasets)
        source_datasets = tuple(d_name for d_name in orders["Source dataset"] if d_name in _all_sou_datasets)

        # ------------
        # Get the best performing models only
        # ------------
        test_scores = extract_selected_best_scores(
            df=results_df, selection_metric=selection_metric, target_metrics=(target_metric,),
            target_datasets=target_datasets, source_datasets=source_datasets
        )
        test_scores_df = pandas.DataFrame(test_scores)

        # test_scores_df.set_index("Source dataset", inplace=True)
        test_scores_df = test_scores_df.pivot(columns="Target dataset", index="Source dataset",
                                              values=f"Performance ({PRETTY_NAME[target_metric]})")

        # Reorder rows and columns
        t_order = tuple(dataset for dataset in orders["Target dataset"] if dataset in target_datasets)
        s_order = tuple(dataset for dataset in orders["Source dataset"] if dataset in source_datasets)

        test_scores_df = test_scores_df.reindex(index=s_order, columns=t_order)

        # Maybe apply the dummy performance
        if metric.subtract_dummy_performance:
            dummy_scores_df = dummy_df.pivot(columns="Target dataset", index="Source dataset",
                                             values=f"Performance ({PRETTY_NAME[target_metric]})")
            dummy_scores_df = dummy_scores_df.reindex(index=s_order, columns=t_order)

            # Subtract such that '+' is improvement to dummy performance
            if higher_is_better(target_metric):
                test_scores_df -= dummy_scores_df
            else:
                test_scores_df = dummy_scores_df - test_scores_df

        # Plotting
        pyplot.figure()
        seaborn.heatmap(
            test_scores_df, annot=True, cbar_kws={"label": PRETTY_NAME[target_metric]},
            fmt=formats[target_metric], vmin=value_ranges[target_metric][0], vmax=value_ranges[target_metric][1],
            **plot_kwargs[target_metric]
        )
        if metric.subtract_dummy_performance:
            pyplot.title("Improvement to dummy baseline")
            # pyplot.title(f"Improvement to dummy baseline (Selection metric: {PRETTY_NAME[selection_metric]})")
        else:
            pyplot.title("Test scores")
            # pyplot.title(f"Test scores (Selection metric: {PRETTY_NAME[selection_metric]})")
        pyplot.savefig(save_path / f"heatmap_{metric.target_metric}.png")
    # pyplot.show()


if __name__ == "__main__":
    main()
