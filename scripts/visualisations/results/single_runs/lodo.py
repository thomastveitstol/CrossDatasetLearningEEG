import os

import pandas
from matplotlib import pyplot

from cdl_eeg.data.paths import get_results_dir


def _get_test_dataset(path, fold):
    # Load the test predictions
    test_pred = pandas.read_csv(os.path.join(path, fold, "test_history_predictions.csv"))  # type: ignore

    # Check the number of datasets in the test set
    if len(set(test_pred["dataset"])) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case for the fold at {os.path.join(path, fold)}. Found "
                         f"{set(test_pred['dataset'])}")

    # If there is a match, return the fold
    return test_pred["dataset"][0]


def _get_metrics(path, fold):
    return pandas.read_csv(os.path.join(path, fold, "test_history_metrics.csv"))  # type: ignore


FONTSIZE = 17
FIGSIZE = (12, 6)
PRETTY_NAME = {"pearson_r": "Pearson's r",
               "spearman_rho": "Spearman's rho",
               "r2_score": r"$R^2$",
               "mae": "MAE",
               "mse": "MSE",
               "mean": "Mean",
               "hatlestad_hall": "HatlestadHall",
               "yulin_wang": "YulinWang",
               "rockhill": "Rockhill",
               "mpi_lemon": "MPI Lemon",
               "miltiadous": "Miltiadous",
               "HatlestadHall": "HatlestadHall",
               "Miltiadous": "Miltiadous",
               "YulinWang": "YulinWang",
               "MPILemon": "MPI Lemon",
               "TDBrain": "TDBRAIN",
               "CAUEEG": "CAUEEG"}


def main():
    # Select run
    run = "debug_age_cv_experiments_2024-04-30_033056"

    # Select metric to plot
    metric = "pearson_r"

    # --------------
    # Get performances
    # --------------
    path = os.path.join(get_results_dir(), run, "leave_one_dataset_out")
    performances = {}
    for fold in (f for f in os.listdir(path) if f[:5] == "Fold_"):
        # Get the dataset name which was the test fold
        dataset = _get_test_dataset(path, fold)

        # Get performance curve
        performances[dataset] = _get_metrics(path, fold)[metric]

    # --------------
    # Plotting
    # --------------
    pyplot.figure(figsize=FIGSIZE)
    for dataset, performance in performances.items():
        pyplot.plot(range(1, len(performance) + 1), performance, label=dataset)

    # Cosmetics
    pyplot.title("LODO test performance", fontsize=FONTSIZE + 5)
    pyplot.ylabel(PRETTY_NAME[metric], fontsize=FONTSIZE)
    pyplot.xlabel("Epoch", fontsize=FONTSIZE)
    pyplot.legend(fontsize=FONTSIZE)
    pyplot.grid()

    pyplot.xticks(fontsize=FONTSIZE)
    pyplot.yticks(fontsize=FONTSIZE)

    pyplot.show()


if __name__ == "__main__":
    main()
