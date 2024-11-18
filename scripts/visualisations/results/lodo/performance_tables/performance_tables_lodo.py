"""
Script for printing and creating performance metric tables
"""
import dataclasses
import os
from typing import Dict

import numpy
import pandas
import torch

from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.analysis.results_analysis import is_better, higher_is_better, get_all_lodo_runs, get_lodo_dataset_name
from cdl_eeg.models.metrics import Histories


# ----------------
# Convenient class
# ----------------
@dataclasses.dataclass(frozen=True)
class _Model:
    path: str  # Path to the results
    test_dataset: str  # The dataset which was used for testing
    val_epoch: int  # The best epoch as estimated on validation set


# -----------
# Test performance estimation
# -----------
def _estimate_intercept(df: pandas.DataFrame):
    # Intercept is calculated as sum(y_i - x_i) / n
    return (df["ground_truth"] - df["pred"]).mean()


def _get_average_prediction(df: pandas.DataFrame, epoch):
    new_df = {"dataset": df["dataset"], "sub_id": df["sub_id"],
              "pred": df.iloc[:, (5 * epoch + 2):(5 * epoch + 7)].mean(axis=1)}
    return pandas.DataFrame.from_dict(data=new_df)


def _get_lodo_test_performance(path, refit_intercept, epoch, target, metrics):
    if not refit_intercept:
        test_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
        test_metrics = {metric: test_df[metric][epoch] for metric in test_df.columns}
        return test_metrics

    test_predictions = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))

    # Check the number of datasets in the test set
    datasets = set(test_predictions["dataset"])
    if len(datasets) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case for the path {path}. Found {set(test_predictions['dataset'])}")
    dataset_name = tuple(datasets)[0]

    # Average the predictions per EEG epoch
    df = _get_average_prediction(test_predictions, epoch=epoch)

    # Add the targets
    df["ground_truth"] = get_dataset(dataset_name).load_targets(target=target, subject_ids=test_predictions["sub_id"])

    # Estimate the intercept  todo: now we are leaking the mean, should we allow this or just the mean of a subset?
    new_intercept = _estimate_intercept(df=df)
    df["adjusted_pred"] = df["pred"] + new_intercept

    # Add the performance
    test_metrics = dict()
    for metric in metrics:
        # Normally, I'd add a 'compute_metric' method to Histories, but I don't like to change the code too much after
        # getting the results from a scientific paper, even when it makes sense. So, violating some best practice
        # instead
        test_metrics[metric] = Histories._compute_metric(
            metric=metric, y_pred=torch.tensor(df["adjusted_pred"]), y_true=torch.tensor(df["ground_truth"])
        )
    return test_metrics

# ----------------
# Functions for getting the results
# ----------------
def _get_lodo_val_metrics_and_epoch(path, *, main_metric, balance_validation_performance):
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
    tuple[float, dict[str, float], str]
        Validation and test set metrics, as well as dataset name
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

    return val_metric, val_idx, get_lodo_dataset_name(path)


def _get_best_lodo_performances(results_dir, *, main_metric, balance_validation_performance, refit_intercept, target,
                                metrics, verbose):
    # Get all runs for LODO
    runs = get_all_lodo_runs(results_dir, successful_only=True)

    # Initialisation
    best_val_metrics: Dict[str, float] = {}
    best_models: Dict[str, _Model] = {}

    # --------------
    # Loop through all experiments
    # --------------
    for run in runs:
        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            # I need to get the validation performance (for making model selection), the best epoch (in case the
            # current run and fold is best for this dataset), and the dataset name (to know which dataset the
            # metrics are for)
            fold_path = os.path.join(run_path, fold)  # type: ignore
            try:
                val_metric, best_epoch, test_dataset = _get_lodo_val_metrics_and_epoch(
                    path=fold_path, main_metric=main_metric,  # type: ignore
                    balance_validation_performance=balance_validation_performance
                )
            except KeyError:
                continue

            # If this is the best run (as evaluated on the validation), store it. (Not the nicest implementation but
            # it'll do)
            if test_dataset not in best_val_metrics or is_better(
                    metric=main_metric, old_metrics={main_metric: best_val_metrics[test_dataset]},
                    new_metrics={main_metric: val_metric}
            ):
                # Update the best model selection
                best_models[test_dataset] = _Model(
                    path=fold_path, test_dataset=test_dataset, val_epoch=best_epoch  # type: ignore
                )

                # Update best metrics
                if test_dataset in best_val_metrics and verbose:
                    print(f"{test_dataset}: {best_val_metrics[test_dataset]:.2f} < {val_metric:.2f}")
                best_val_metrics[test_dataset] = val_metric

    # --------------
    # Get results to a dataframe
    # --------------
    # Sort the dict first
    best_models = {dataset: best_models[dataset] for dataset in DATASET_ORDER}

    results = {"target_dataset": [], **{metric: [] for metric in metrics}}
    for dataset, model in best_models.items():
        assert dataset == model.test_dataset

        best_run = model.path.split("/")[-3]
        print(f"Best run ({dataset}): {best_run}")
        test_performance = _get_lodo_test_performance(
            path=model.path, refit_intercept=refit_intercept, epoch=model.val_epoch, target=target, metrics=metrics
        )
        results["target_dataset"].append(dataset)

        for metric, performance in test_performance.items():
            results[metric].append(performance)

    return pandas.DataFrame.from_dict(results).round(DECIMALS)


# -------------
# Globals
# -------------
DECIMALS = 2
DATASET_ORDER = ("TDBrain", "MPILemon", "HatlestadHall", "Miltiadous", "YulinWang")


def main():
    # Hyperparameters
    selection_metrics = ("mae", "mse", "pearson_r", "r2_score")
    all_metrics = ("mae", "mse", "mape", "pearson_r", "spearman_rho", "r2_score")
    balance_validation_performance = False
    target = "age"
    verbose = False

    # Print results
    for refit_intercept in (True, False):
        for selection_metric in selection_metrics:
            print(f"\n\n{f' Selection metric: {selection_metric} ':=^50}\n")
            df = _get_best_lodo_performances(
                results_dir=get_results_dir(), main_metric=selection_metric, refit_intercept=refit_intercept,
                balance_validation_performance=balance_validation_performance, target=target, metrics=all_metrics,
                verbose=verbose
            )
            print(df)

            # Save the results
            df.to_csv(os.path.join(os.path.dirname(__file__), f"results_{selection_metric}_refit_intercept_"
                                                              f"{refit_intercept}.csv"))


if __name__ == "__main__":
    main()
