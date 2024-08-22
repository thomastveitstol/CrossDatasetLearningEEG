import os
from typing import Dict, List, NamedTuple

import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.results_analysis import higher_is_better, SkipFold, get_all_lodo_runs, PRETTY_NAME, \
    get_lodo_dataset_name


# ----------------
# Convenient small classes
# ----------------
class ValTestPerformances(NamedTuple):
    val: float
    test: Dict[str, float]


# ----------------
# Functions for getting the results
# ----------------
def _get_validation_performance(path, *, main_metric, balance_validation_performance):
    # Input check
    if not isinstance(balance_validation_performance, bool):
        raise TypeError(f"Expected argument 'balance_validation_performance' to be boolean, but found "
                        f"{type(balance_validation_performance)}")

    # Get the best epoch, as evaluated on the validation set
    if balance_validation_performance:
        val_df_path = os.path.join(path, "sub_groups_plots", "dataset_name", main_metric, f"val_{main_metric}.csv")
        val_df = pandas.read_csv(val_df_path)  # type: ignore

        val_performances = numpy.mean(val_df.values, axis=-1)

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_performances)
        else:
            val_idx = numpy.argmin(val_performances)

        # Currently, we only actually need the 'main_metric'
        val_metric = val_performances[val_idx]

    else:
        # Load the dataframe of the validation performances
        val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

        # Get the best performance and its epoch
        if higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_df[main_metric])
        else:
            val_idx = numpy.argmin(val_df[main_metric])
        val_metric = val_df[main_metric][val_idx]

    return val_metric, val_idx


def _get_test_val_metrics(path,  *, main_metric, datasets, balance_validation_performance):
    # --------------
    # Get test dataset name
    # --------------
    dataset_name = get_lodo_dataset_name(path)
    if dataset_name not in datasets:
        # If we are not interested, we will not waste time loading data for then to discard the results
        raise SkipFold

    # --------------
    # Get validation performance and optimal epoch
    # --------------
    val_performance, epoch = _get_validation_performance(path=path, main_metric=main_metric,
                                                         balance_validation_performance=balance_validation_performance)

    # --------------
    # Get test performance
    # --------------
    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_performance = {metric: test_df[metric][epoch] for metric in test_df.columns}

    return val_performance, test_performance, dataset_name


def plot_test_vs_val_lodo(results_dir, *, main_metric, metrics_to_plot, datasets, balance_validation_performance, x_lim,
                          y_lims):
    # Get all runs for LODO
    runs = get_all_lodo_runs(results_dir)

    # Initialisation
    performances: Dict[str, List[ValTestPerformances]] = {dataset: [] for dataset in datasets}

    # --------------
    # Loop through all experiments
    # --------------
    num_runs = len(runs)
    skipped = {dataset: 0 for dataset in datasets}
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(f"Run {i + 1}/{num_runs}")

        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            try:
                val_metric, test_metrics, test_dataset = _get_test_val_metrics(
                    path=os.path.join(run_path, fold), main_metric=main_metric, datasets=datasets,  # type: ignore
                    balance_validation_performance=balance_validation_performance
                )
            except SkipFold:
                continue
            except KeyError:
                # If the prediction model guessed that all subjects have the same age, for all folds, model selection
                # 'fails'. We'll just skip them
                skipped[get_lodo_dataset_name(os.path.join(run_path, fold))] += 1  # type: ignore
                continue

            # Add the val and test performances
            performances[test_dataset].append(ValTestPerformances(val=val_metric, test=test_metrics))

    print(f"Skipped runs: {skipped}")

    # --------------
    # Plot the results
    # --------------
    for test_dataset, val_test_performances in performances.items():
        # Get the validation and test performances
        x = [performance.val for performance in val_test_performances]
        test_performance = [performance.test for performance in val_test_performances]

        # The test performances must be plotted per metric
        for metric in metrics_to_plot:
            pyplot.figure(figsize=FIGSIZE)

            y = [performance[metric] for performance in test_performance]

            pyplot.plot(x, y, ".")

            # Plot cosmetics
            pyplot.title(f"Target dataset: {PRETTY_NAME[test_dataset]}", fontsize=FONTSIZE + 5)
            pyplot.ylabel(f"Test performance ({PRETTY_NAME[metric]})", fontsize=FONTSIZE)
            pyplot.xlabel(f"Validation performance ({PRETTY_NAME[main_metric]})", fontsize=FONTSIZE)
            pyplot.tick_params(labelsize=FONTSIZE)
            pyplot.xlim(x_lim)
            pyplot.ylim(y_lims[metric])
            pyplot.grid()
            pyplot.tight_layout()

    pyplot.show()


# ----------------
# Globals
# ----------------
FIGSIZE = (7, 5)
FONTSIZE = 16


def main():
    # Hyperparameters
    main_metric = "pearson_r"
    metrics_to_plot = ("pearson_r",)
    datasets = ("HatlestadHall", "MPILemon", "TDBrain", "Miltiadous", "YulinWang")
    balance_validation_performance = False

    x_lims = {"pearson_r": (-0.25, 1), "mae": (0, None), "r2_score": (None, 1)}  # Limits selected post-hoc
    y_lims = {"pearson_r": (-0.45, 1), "mae": (0, None), "r2_score": (None, 1)}  # Limits selected post-hoc
    x_lim = x_lims[main_metric]

    # Plot results
    plot_test_vs_val_lodo(results_dir=get_results_dir(), main_metric=main_metric, metrics_to_plot=metrics_to_plot,
                          datasets=datasets, balance_validation_performance=balance_validation_performance, x_lim=x_lim,
                          y_lims=y_lims)


if __name__ == "__main__":
    main()
