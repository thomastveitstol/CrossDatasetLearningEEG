import os
import pathlib

import pandas

from cdl_eeg.data.analysis.results_analysis import get_all_lodo_runs, SkipFold, get_lodo_test_performance, \
    get_lodo_dataset_name, PRETTY_NAME, get_all_ilodo_runs, get_lodi_test_performance, get_ilodo_val_dataset_name
from cdl_eeg.data.paths import get_results_dir


def _generate_dataframe(results_dir, *, target_metrics, balance_validation_performance, datasets, selection_metric,
                        refit_intercept):
    # Initialise dict which will be used for plotting
    results = {"Target dataset": [], "Source dataset": [], "run": [], **{metric: [] for metric in target_metrics}}

    # --------------
    # Obtain all test results from the LODO runs
    # --------------
    # Get all runs for LODO
    lodo_runs = get_all_lodo_runs(results_dir, successful_only=True)
    num_runs = len(lodo_runs)
    skipped = {dataset: 0 for dataset in datasets}
    for i, run in enumerate(lodo_runs):
        if i % 10 == 0:  # Not the best way...
            print(f"Run {i + 1}/{num_runs}")

        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            try:
                test_performance, test_dataset = get_lodo_test_performance(
                    path=os.path.join(run_path, fold), selection_metric=selection_metric,  # type: ignore
                    datasets=datasets, balance_validation_performance=balance_validation_performance,
                    target_metrics=target_metrics
                )
            except SkipFold:
                continue
            except KeyError:
                # If the prediction model guessed that all subjects have the same age, for all folds, model selection
                # 'fails'. We'll just skip them
                skipped[get_lodo_dataset_name(os.path.join(run_path, fold))] += 1  # type: ignore
                continue

            # Add to test performances
            results["Target dataset"].append(PRETTY_NAME[test_dataset])
            results["Source dataset"].append("Pooled")
            results["run"].append(run)
            for metric, performance in test_performance.items():
                results[metric].append(performance)

    # --------------
    # Obtain all test results from the LODI runs
    # --------------
    lodi_runs = get_all_ilodo_runs(results_dir, successful_only=True)
    num_runs = len(lodi_runs)
    skipped = {dataset: 0 for dataset in datasets}
    for i, run in enumerate(lodi_runs):
        if i % 10 == 0:  # Not the best way...
            print(f"Run {i + 1}/{num_runs}")

        run_path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Get the performances per fold
        folds = (path for path in os.listdir(run_path) if path[:5] == "Fold_")  # type: ignore
        for fold in folds:
            try:
                test_performance, train_dataset = get_lodi_test_performance(
                    path=os.path.join(run_path, fold), selection_metric=selection_metric,  # type: ignore
                    datasets=datasets, target_metrics=target_metrics, refit_intercept=refit_intercept
                )
            except SkipFold:
                continue
            except KeyError:
                # If the prediction model guessed that all subjects have the same age, for all folds, model selection
                # 'fails'. We'll just skip them
                skipped[get_ilodo_val_dataset_name(os.path.join(run_path, fold))] += 1  # type: ignore
                continue

            # Add to test performances
            for test_dataset_name, performance_scores in test_performance.items():
                results["Target dataset"].append(PRETTY_NAME[test_dataset_name])
                results["Source dataset"].append(PRETTY_NAME[train_dataset])
                results["run"].append(run)
                for metric, score in performance_scores.items():
                    results[metric].append(score)

    return pandas.DataFrame(results)


def main():
    # -------------
    # Some decisions to make for generating the dataframe
    # -------------
    results_dir = get_results_dir()
    target_metrics = ("pearson_r", "spearman_rho", "r2_score", "mae", "mse", "mape")
    datasets = ("TDBrain", "MPILemon", "HatlestadHall", "Miltiadous", "YulinWang")
    selection_metric = "pearson_r"
    balance_validation_performance = False
    refit_intercept = False

    df_name = f"all_test_results_selection_metric_{selection_metric}"
    save_to_path = pathlib.Path(os.path.dirname(__file__))

    # -------------
    # Generate and save dataframe
    # -------------
    df = _generate_dataframe(
        results_dir, target_metrics=target_metrics, balance_validation_performance=balance_validation_performance,
        datasets=datasets, selection_metric=selection_metric, refit_intercept=refit_intercept
    )

    df.to_csv((save_to_path / df_name).with_suffix(".csv"))



if __name__ == "__main__":
    main()
