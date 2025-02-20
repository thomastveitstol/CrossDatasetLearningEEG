import os
import pathlib

import pandas

from cdl_eeg.data.analysis.results_analysis import get_all_lodo_runs, SkipFold, get_lodo_test_performance, \
    PRETTY_NAME, get_all_ilodo_runs, get_lodi_test_performance
from cdl_eeg.data.paths import get_results_dir


def _generate_dataframe(results_dir, *, target_metrics, balance_validation_performance, datasets, selection_metric,
                        refit_metrics):
    # Initialise dict which will be used for plotting
    results = {"Target dataset": [], "Source dataset": [], "run": [], "Val score": [],
               **{metric: [] for metric in target_metrics}, **{f"{metric}_refit": [] for metric in refit_metrics}}

    # --------------
    # Obtain all test results from the LODO runs
    # --------------
    # Get all runs for LODO
    lodo_runs = get_all_lodo_runs(results_dir, successful_only=True)
    try:
        from progressbar import progressbar  # type: ignore
        lodo_loop = progressbar(lodo_runs, redirect_stdout=True, prefix="Run ")
        using_progressbar = True
    except ImportError:
        lodo_loop = lodo_runs
        using_progressbar = False
    for run in lodo_loop:
        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            try:
                test_performance, test_dataset, val_performance = get_lodo_test_performance(
                    path=os.path.join(run_path, fold), selection_metric=selection_metric,  # type: ignore
                    datasets=datasets, balance_validation_performance=balance_validation_performance,
                    target_metrics=target_metrics, refit_metrics=refit_metrics
                )
            except SkipFold:
                continue

            # Add to test performances
            results["Target dataset"].append(PRETTY_NAME[test_dataset])
            results["Source dataset"].append("Pooled")
            results["run"].append(run)
            results["Val score"].append(val_performance)
            for metric, performance in test_performance.items():
                if not isinstance(performance, float):
                    raise TypeError(f"Expected performance score to be float, but found ({type(performance)}): "
                                    f"{performance}")
                results[metric].append(performance)

    # --------------
    # Obtain all test results from the LODI runs
    # --------------
    lodi_runs = get_all_ilodo_runs(results_dir, successful_only=True)
    if using_progressbar:
        # noinspection PyUnboundLocalVariable
        lodi_loop = progressbar(lodi_runs, redirect_stdout=True, prefix="Run ")
    else:
        lodi_loop = lodi_runs
    for run in lodi_loop:
        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            try:
                test_performance, train_dataset, val_performance = get_lodi_test_performance(
                    path=os.path.join(run_path, fold), selection_metric=selection_metric,  # type: ignore
                    datasets=datasets, target_metrics=target_metrics, refit_metrics=refit_metrics
                )
            except SkipFold:
                continue

            # Add to test performances
            for test_dataset_name, performance_scores in test_performance.items():
                results["Target dataset"].append(PRETTY_NAME[test_dataset_name])
                results["Source dataset"].append(PRETTY_NAME[train_dataset])
                results["run"].append(run)
                results["Val score"].append(val_performance)
                for metric, score in performance_scores.items():
                    if not isinstance(score, float):
                        raise TypeError(f"Expected performance score to be float, but found ({type(score)}): "
                                        f"{score}")
                    results[metric].append(score)

    return pandas.DataFrame(results)


def main():
    # -------------
    # Some decisions to make for generating the dataframe
    # -------------
    results_dir = get_results_dir()
    target_metrics = ("pearson_r", "spearman_rho", "r2_score", "mae", "mse", "mape")
    datasets = ("TDBrain", "MPILemon", "HatlestadHall", "Miltiadous", "YulinWang")
    selection_metrics = ("pearson_r", "spearman_rho", "r2_score", "mae", "mse", "mape")
    balance_validation_performance = False
    refit_metrics = ("r2_score", "mae", "mse")

    # -------------
    # Generate and save dataframe
    # -------------
    for selection_metric in selection_metrics:
        df = _generate_dataframe(
            results_dir, target_metrics=target_metrics, balance_validation_performance=balance_validation_performance,
            datasets=datasets, selection_metric=selection_metric, refit_metrics=refit_metrics
        )

        df_name = f"all_test_results_selection_metric_{selection_metric}"
        save_to_path = pathlib.Path(os.path.dirname(__file__))

        df.to_csv((save_to_path / df_name).with_suffix(".csv"))


if __name__ == "__main__":
    main()
