"""
Script for plotting predictions
"""
import os.path
import re
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import PRETTY_NAME
from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_results_dir
from scripts.visualisations.results.lodo.performance_tables.performance_tables_lodo import get_average_prediction


def _sanity_check_test_predictions(test_predictions, expected_dataset):
    # Check the number of datasets in the test set
    datasets = set(test_predictions["dataset"])
    if len(datasets) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case. Found {set(test_predictions['dataset'])}")

    loaded_dataset = tuple(datasets)[0]
    if loaded_dataset != expected_dataset:
        raise ValueError(f"Expected dataset {expected_dataset}, but found {loaded_dataset}")


def _read_dataset_and_epoch(line):
    # Get the dataset
    dataset = re.findall(r'\((.*?)\)', line)
    assert len(dataset) == 1
    dataset = dataset[0]

    # Get the epoch
    epoch = re.findall(r'\|(.*?)\|', line)
    assert len(epoch) == 1
    epoch = int(epoch[0].split(" ")[-1])

    return dataset, epoch


def _get_predictions(dataset_name, path, epoch):
    # Load csv file
    test_predictions = pandas.read_csv(os.path.join(path, "test_history_predictions.csv"))
    _sanity_check_test_predictions(test_predictions, expected_dataset=dataset_name)

    # Aggregate predictions
    df = get_average_prediction(test_predictions, epoch=epoch)

    # Add the targets
    df["ground_truth"] = get_dataset(dataset_name).load_targets(target=TARGET, subject_ids=test_predictions["sub_id"])

    return df[["pred", "ground_truth"]]


TARGET = "age"

FIGSIZE = (6.7, 6.7)
FONTSIZE = 18
TITLE_FONTSIZE = FONTSIZE + 3
X_LIM = (0, 90)
Y_LIM = (0, 90)
DPI = 300
GRID_SPACING = 10


def main():
    # --------------
    # Design choices
    # --------------
    datasets = ("TDBrain", "MPILemon", "HatlestadHall", "Miltiadous", "YulinWang")
    selection_metrics = ("mae", "mse", "pearson_r", "r2_score")
    refit_intercept = False

    # --------------
    # Make plots
    # --------------
    # Path
    models_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "performance_tables"  # Best models are stored here
    results_dir = Path(get_results_dir())  # The results are stored here

    # Loop through
    for metric in selection_metrics:
        # Load the details of the selected best models
        with open((models_dir / f"{metric}_refit_intercept_{refit_intercept}").with_suffix(".txt"), "r") as file:
            for line in file.readlines():
                # --------------
                # Get the data to plot
                # --------------
                # Get the dataset and epoch
                dataset, epoch = _read_dataset_and_epoch(line=line)
                if dataset not in datasets:
                    continue

                # Get the predictions and ground truth
                configuration_path = line.replace("\n", "").split(" ")[-1]
                df = _get_predictions(dataset_name=dataset, path=results_dir / configuration_path, epoch=epoch)

                # --------------
                # Plotting
                # --------------
                # Plot
                pyplot.figure(figsize=FIGSIZE)
                pyplot.plot(df["ground_truth"], df["pred"], ".")
                pyplot.plot(X_LIM, Y_LIM, color="black")  # Plot line

                # Plot cosmetics
                pyplot.xlim(X_LIM)
                pyplot.ylim(Y_LIM)
                pyplot.xticks(numpy.arange(*X_LIM, GRID_SPACING))
                pyplot.yticks(numpy.arange(*Y_LIM, GRID_SPACING))
                pyplot.title(f"Selection metric: {PRETTY_NAME[metric]}", fontsize=TITLE_FONTSIZE)
                pyplot.tick_params(labelsize=FONTSIZE)
                if dataset == datasets[-1]:
                    pyplot.xlabel("Ground truth", fontsize=FONTSIZE)
                else:
                    pyplot.tick_params(bottom=False, labelbottom=False)
                if metric == selection_metrics[0]:
                    pyplot.ylabel("Predicted", fontsize=FONTSIZE)
                else:
                    pyplot.tick_params(left=False, labelleft=False)
                pyplot.grid()


                # Save the figure
                _to_path = os.path.join(
                    os.path.dirname(__file__), "plots",
                    f"predictions_lodo_{PRETTY_NAME[dataset].lower()}_selection_metric_{metric}"
                )
                pyplot.savefig(_to_path, dpi=DPI)


if __name__ == "__main__":
    main()
