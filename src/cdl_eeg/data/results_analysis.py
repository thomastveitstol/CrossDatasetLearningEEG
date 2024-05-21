import os
from typing import Dict, List

import numpy
import pandas


def _higher_is_better(metric):
    if metric in ("pearson_r", "spearman_rho", "r2_score"):
        return True
    elif metric in ("mae", "mse", "mape"):
        return False
    else:
        raise ValueError(f"Metric {metric} not recognised")


def _is_better(metric, *, old_metrics, new_metrics):
    # Input checks
    assert isinstance(metric, str), f"Expected metric to be string, but found {type(metric)}"
    assert isinstance(old_metrics, dict), f"Expected 'old_metrics' to be dict, but found {type(old_metrics)}"
    assert isinstance(new_metrics, dict), f"Expected 'new_metrics' to be dict, but found {type(new_metrics)}"

    # If old metrics is and empty dict, the new ones are considered the best
    if not old_metrics:
        return True

    # Return
    if _higher_is_better(metric=metric):
        return new_metrics[metric] > old_metrics[metric]
    else:
        return new_metrics[metric] < old_metrics[metric]


def _get_all_lodo_runs(results_dir):
    """Function for getting all runs available for leave-one-dataset-out"""
    return (run for run in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, run,
                                                                                  "leave_one_dataset_out",
                                                                                  "finished_successfully.txt"))
            and "inverted_cv" not in run)


def _get_lodo_dataset_name(path) -> str:
    """Function for getting the name of the test dataset. A test is also made to ensure that the test set only contains
    one dataset"""
    # Load the test predictions
    test_df = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))

    # Check the number of datasets in the test set
    if len(set(test_df["dataset"])) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case for the path {path}. Found {set(test_df['dataset'])}")

    # Return the dataset name
    return test_df["dataset"][0]


def _get_lodo_val_test_metrics(path, *, main_metric, balance_validation_performance):
    """
    Function for getting the validation and test metrics from a single fold. The epoch is selected based on validation
    set performance

    Parameters
    ----------
    path : str
    main_metric : str
        The metric which determines what is 'best'
    balance_validation_performance : bool
        If True, the 'best' performance is computed as the average on the datasets. Otherwise, it is computed on a pool
        of all the data

    Returns
    -------
    tuple[dict[str, float], dict[str, float], str]
        Validation and test set metrics
    """
    # Input check
    if not isinstance(balance_validation_performance, bool):
        raise TypeError(f"Expected argument 'balance_validation_performance' to be boolean, but found "
                        f"{type(balance_validation_performance)}")

    # Get the best epoch, as evaluated on the validation set
    if balance_validation_performance:
        val_df_path = os.path.join(path, "sub_groups_plots", "dataset_name", main_metric, f"val_{main_metric}.csv")
        val_df = pandas.read_csv(val_df_path)

        val_performances = numpy.mean(val_df.values, axis=-1)

        # Get the best performance and its epoch
        if _higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_performances)
        else:
            val_idx = numpy.argmin(val_performances)

        # Currently, we only actually need the 'main_metric'
        val_metrics = {main_metric: val_performances[val_idx]}

    else:
        # Load the dataframe of the validation performances
        val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

        # Get the best performance and its epoch
        if _higher_is_better(metric=main_metric):
            val_idx = numpy.argmax(val_df[main_metric])
        else:
            val_idx = numpy.argmin(val_df[main_metric])

        val_metrics = {main_metric: val_df[metric][val_idx] for metric in val_df.columns}

    # Return the validation and test performances from the same epoch, as well as the name of the test dataset
    test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))

    test_metrics = {metric: test_df[metric][val_idx] for metric in test_df.columns}
    return val_metrics, test_metrics, _get_lodo_dataset_name(path)


def get_best_lodo_performances(results_dir, *, main_metric, balance_validation_performance):
    """Function for getting the metrics of the highest performing model"""
    # Get all runs for LODO
    runs = _get_all_lodo_runs(results_dir)

    # Some initialisation
    best_val_metrics: Dict[str, float] = dict()
    best_test_fold_performances: Dict[str, Dict[str, float]] = dict()
    best_run: str = ""

    # --------------
    # Loop through all experiments
    # --------------
    for run in runs:
        try:
            run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

            val_fold_performances: Dict[str, List[float]] = dict()
            test_fold_performances: Dict[str, Dict[str, float]] = dict()

            # Get the performances per fold
            folds = tuple(path for path in os.listdir(run_path) if path[:5] == "Fold_")
            for i, fold in enumerate(folds):
                # Get the test metrics for the current fold. The epoch was selected by maximising validation set
                # performance
                val_metrics, test_metrics, test_dataset = _get_lodo_val_test_metrics(
                    path=os.path.join(run_path, fold), main_metric=main_metric,
                    balance_validation_performance=balance_validation_performance
                )

                # Store validation fold performance
                for metric, performance in val_metrics.items():
                    if i == 0:
                        val_fold_performances[metric] = [performance]
                    else:
                        val_fold_performances[metric].append(performance)

                # Store test fold performance
                test_fold_performances[test_dataset] = dict()
                for metric, performance in test_metrics.items():
                    test_fold_performances[test_dataset][metric] = performance

            # If this is the best run (as evaluated on the validation set), store validation and test performances
            avg_val_metrics: Dict[str, float] = {metric: numpy.mean(performances)
                                                 for metric, performances in val_fold_performances.items()}
            if _is_better(metric=main_metric, old_metrics=best_val_metrics, new_metrics=avg_val_metrics):
                best_run = run
                best_val_metrics = avg_val_metrics
                best_test_fold_performances = test_fold_performances
        except KeyError:
            continue

    # --------------
    # Print results
    # --------------
    print(f"The best run: {best_run}\n")
    print("Test scores:")
    for dataset, metrics in best_test_fold_performances.items():
        print(f"{f' {dataset} ':-^20}")
        for metric, performance in metrics.items():
            print(f"\t{metric}: {performance:.2f}")
