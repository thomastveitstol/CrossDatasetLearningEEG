import os

import optuna
import yaml

from cdl_eeg.data.analysis.hyperparameter_importance import create_studies
from cdl_eeg.data.analysis.results_analysis import get_all_lodo_runs, higher_is_better, PRETTY_NAME
from cdl_eeg.data.paths import get_results_dir


def main():
    # ----------------
    # A few design choices for the analysis
    # ----------------
    datasets = ("TDBrain", "MPILemon", "HatlestadHall")
    results_dir = get_results_dir()
    selection_metric = "mae"
    target_metric = "pearson_r"
    direction = "maximize" if higher_is_better(target_metric) else "minimize"
    balance_validation_performance = False
    hyperparameters = None
    runs = get_all_lodo_runs(results_dir=results_dir, successful_only=True)
    config_dist_path = os.path.join(  # todo: not very elegant...
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "training",
        "config_files", "hyperparameter_random_search.yml"
    )
    with open(config_dist_path) as f:
        config_dist = yaml.safe_load(f)

    # ----------------
    # Create the studies
    # ----------------
    studies = create_studies(
        datasets=datasets, runs=runs, direction=direction, results_dir=results_dir, target_metric=target_metric,
        selection_metric=selection_metric, balance_validation_performance=balance_validation_performance,
        hyperparameters=hyperparameters, dist_config=config_dist
    )

    # ----------------
    # Analyse the hyperparameters
    # ----------------
    for dataset_name, study in studies.items():
        print(f"=== {PRETTY_NAME[dataset_name]} ===")
        for param_name, param_value in study.best_params.items():
            print(f"{param_name}: {param_value}")

        # Plotting
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()


if __name__ == "__main__":
    main()
