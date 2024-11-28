import dataclasses
import os
import sys
from typing import Dict

import numpy
import pandas
import torch

from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.analysis.results_analysis import get_all_ilodo_runs, higher_is_better, is_better, \
    get_ilodo_val_dataset_name, PRETTY_NAME
from cdl_eeg.models.metrics import Histories


# ----------------
# Convenient class
# ----------------
@dataclasses.dataclass(frozen=True)
class _Model:
    path: str  # Absolute path to the results
    train_dataset: str  # The dataset which was used for training
    best_epoch: int  # The best epoch


# ----------------
# Functions for getting the results
# ----------------
def _get_lodo_val_metrics(path, *, main_metric):
    """
    Function for getting the best validation metric for a single fold. It also returns the best epoch and the name of
    the trainig dataset

    Parameters
    ----------
    path : str
    main_metric : str

    Returns
    -------
    tuple[float, int, str]
        Validation performance, the best epoch, and train dataset name
    """
    # --------------
    # Get training dataset name
    # --------------
    dataset_name = get_ilodo_val_dataset_name(path)

    # --------------
    # Get validation performance and optimal epoch
    # --------------
    # Load the dataframe of the validation performances
    val_df = pandas.read_csv(os.path.join(path, "val_history_metrics.csv"))

    # Get the best performance and its epoch
    if higher_is_better(metric=main_metric):
        best_epoch = numpy.argmax(val_df[main_metric])
    else:
        best_epoch = numpy.argmin(val_df[main_metric])
    val_metric = val_df[main_metric][best_epoch]

    return val_metric, best_epoch, dataset_name


# ----------------
# Functions for refitting intercept
# ----------------
def _estimate_intercept(df: pandas.DataFrame):
    # Intercept is calculated as sum(y_i - x_i) / n
    return (df["ground_truth"] - df["pred"]).mean()


def _get_average_prediction(df: pandas.DataFrame, epoch):
    new_df = {"dataset": df["dataset"], "sub_id": df["sub_id"],
              "pred": df.iloc[:, (5 * epoch + 2):(5 * epoch + 7)].mean(axis=1)}
    return pandas.DataFrame.from_dict(data=new_df)


def _get_ilodo_refit_scores(path, epoch, metrics) -> Dict[str, Dict[str, float]]:
    test_predictions = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))
    test_metrics = dict()

    # ------------
    # Loop through all datasets
    # ------------
    datasets = set(test_predictions["dataset"])
    for dataset in datasets:
        # Get the predictions for this dataset only
        df = test_predictions[test_predictions["dataset"] == dataset].copy()

        # Average the predictions per EEG epoch
        df = _get_average_prediction(df=df, epoch=epoch)

        # Add the targets (age)
        target = "age"  # quick-fixed hard coding
        df["ground_truth"] = get_dataset(dataset).load_targets(target=target, subject_ids=df["sub_id"])

        # Estimate the intercept
        new_intercept = _estimate_intercept(df=df)
        df["adjusted_pred"] = df["pred"] + new_intercept

        # Add the performance
        test_metrics[dataset] = dict()
        for metric in metrics:
            # Normally, I'd add a 'compute_metric' method to Histories, but I don't like to change the code too much after
            # getting the results from a scientific paper, even when it makes sense. So, violating some best practice
            # instead
            test_metrics[dataset][metric] = Histories._compute_metric(
                metric=metric, y_pred=torch.tensor(df["adjusted_pred"]), y_true=torch.tensor(df["ground_truth"])
            )

    # ------------
    # Now, fix the pooled dataset
    # ------------
    df = test_predictions

    # Average the predictions per EEG epoch
    df = _get_average_prediction(df=df, epoch=epoch)

    # Sorting makes life easier
    df.sort_values(by=["dataset"], inplace=True)

    # Add the targets (age)
    target = "age"  # quick-fixed hard coding
    ground_truths = []
    for dataset in datasets:
        ground_truths.append(
            get_dataset(dataset).load_targets(target=target, subject_ids=df["sub_id"][df["dataset"] == dataset])
        )
    df["ground_truth"] = numpy.concatenate(ground_truths)
    new_intercept = _estimate_intercept(df=df)
    df["adjusted_pred"] = df["pred"] + new_intercept

    # Add the performance
    test_metrics["Pooled"] = dict()
    for metric in metrics:
        test_metrics["Pooled"][metric] = Histories._compute_metric(
            metric=metric, y_pred=torch.tensor(df["adjusted_pred"]), y_true=torch.tensor(df["ground_truth"])
        )
    return test_metrics


def _get_ilodo_test_metrics(path, epoch, refit_intercept):
    # -----------------
    # Get the test metrics per test dataset
    # -----------------
    # Get path to where the metrics are stored
    subgroup_path = os.path.join(path, "sub_groups_plots", "dataset_name")  # hard-coded for now
    metrics = os.listdir(subgroup_path)

    # Get all metrics
    if refit_intercept:
        test_metrics = _get_ilodo_refit_scores(path, epoch, metrics)
    else:
        test_metrics = {}
        for metric in metrics:
            df = pandas.read_csv(os.path.join(subgroup_path, metric, f"test_{metric}.csv"))

            # Loop through all the test datasets
            datasets = df.columns
            for dataset in datasets:
                if dataset not in test_metrics:
                    test_metrics[dataset] = {}

                # Add the test performance
                test_metrics[dataset][metric] = df[dataset][epoch]

    # -----------------
    # Add the test metrics on the pooled dataset
    # -----------------
    pooled_df = pandas.read_csv(os.path.join(path, "test_history_metrics.csv"))
    test_metrics["Pooled"] = {metric: pooled_df[metric][epoch] for metric in pooled_df.columns}

    return test_metrics


def get_best_ilodo_performances(results_dir, *, main_metric, metrics_to_print, verbose, refit_intercept):
    # Get all runs for inverted LODO
    runs = get_all_ilodo_runs(results_dir)

    # Initialisation
    best_val_metrics: Dict[str, float] = {}
    best_models: Dict[str, _Model] = {}

    # --------------
    # Loop through all experiments for model selection
    # --------------
    for run in runs:
        try:
            run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

            # Get the performances per fold
            folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
            for fold in folds:
                # I need the validation performance and the dataset which was used. The test set performances is not
                # acquired here to reduce runtime
                val_metric, epoch, train_dataset = _get_lodo_val_metrics(
                    path=os.path.join(run_path, fold), main_metric=main_metric  # type: ignore
                )

                # If this is the best run (as evaluated on the validation), store the details
                if train_dataset not in best_val_metrics or is_better(
                        metric=main_metric, old_metrics={main_metric: best_val_metrics[train_dataset]},
                        new_metrics={main_metric: val_metric}
                ):
                    # Update the best model selection
                    best_models[train_dataset] = _Model(path=os.path.join(run_path, fold),  # type: ignore
                                                        train_dataset=train_dataset,
                                                        best_epoch=epoch)

                    # Update best metrics
                    if train_dataset in best_val_metrics and verbose:
                        print(f"{train_dataset}: {best_val_metrics[train_dataset]:.2f} < {val_metric:.2f}")
                    best_val_metrics[train_dataset] = val_metric
        except KeyError:
            continue

    # --------------
    # Get the test performances from the best models
    # --------------
    # E.g., {"TDBrain": {"LEMON": {"mse": 10.4, "mae": 2.7}}}
    test_performances: Dict[str, Dict[str, Dict[str, float]]] = {}
    for train_dataset, model in best_models.items():
        test_performances[train_dataset] = _get_ilodo_test_metrics(path=model.path, epoch=model.best_epoch,
                                                                   refit_intercept=refit_intercept)

    # --------------
    # Generate dataframe from the results
    # --------------
    print(f"{' Inverted LODO ':=^30}")

    # Initialise dict for storing results
    _dataset_order = tuple(PRETTY_NAME[dataset] for dataset in DATASET_ORDER + ("Pooled",))
    results = {metric: {"source_dataset": [], **{dataset: [] for dataset in _dataset_order}}
               for metric in metrics_to_print}

    # Order the results properly
    test_performances = {dataset_name: test_performances[dataset_name] for dataset_name in DATASET_ORDER}

    # Loop through
    for train_dataset, generalisation_performances in test_performances.items():
        model = best_models[train_dataset]

        print(f"Best run ({train_dataset}): {model.path.split('/')[-3]}")
        for metric in metrics_to_print:
            results[metric]["source_dataset"].append(PRETTY_NAME[train_dataset])
            for test_dataset in DATASET_ORDER + ("Pooled",):
                _pretty_test_name = PRETTY_NAME[test_dataset]
                if test_dataset == train_dataset:
                    results[metric][_pretty_test_name].append(numpy.nan)
                else:
                    results[metric][_pretty_test_name].append(generalisation_performances[test_dataset][metric])

    dfs = {metric: pandas.DataFrame(results_table).round(DECIMALS) for metric, results_table in results.items()}
    df = pandas.concat(dfs, names=("metric", "original_index")).reset_index("original_index", drop=True)

    return df, dfs


# -------------
# Globals
# -------------
DECIMALS = 2
DATASET_ORDER = ("TDBrain", "MPILemon", "HatlestadHall", "Miltiadous", "YulinWang")


def main():
    # Hyperparameters
    selection_metrics = ("mae", "mse", "pearson_r", "r2_score")
    all_metrics = ("mae", "mse", "mape", "pearson_r", "spearman_rho", "r2_score")

    verbose = False

    # Print results
    for refit_intercept in (True, False):
        for selection_metric in selection_metrics:
            print(f"\n\n{f' Selection metric: {selection_metric} ':=^50}\n")
            df, dfs = get_best_ilodo_performances(
                results_dir=get_results_dir(), main_metric=selection_metric, metrics_to_print=all_metrics,
                verbose=verbose, refit_intercept=refit_intercept
            )

            # Print the results for easy copy/paste into overleaf document (some changes are still needed, but it still
            # speeds up the process and is less error-prone)
            for metric, results_table in dfs.items():
                print("\hline")
                print(f"\multirow{r'{5}'}{r'{*}'}{r'{'}{PRETTY_NAME[metric]}{r'}'}")
                results_table.to_csv(sys.stdout, sep="&", header=False)

            print("\hline")

            # Save the results  todo: must add refitting of intercept
            df.to_csv(os.path.join(os.path.dirname(__file__),
                                   f"results_{selection_metric}_intercept_{refit_intercept}.csv"))


if __name__ == "__main__":
    main()
