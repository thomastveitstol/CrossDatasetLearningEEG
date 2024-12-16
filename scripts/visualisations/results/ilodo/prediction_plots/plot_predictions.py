"""
Script for plotting predictions
"""
import os
import re
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import PRETTY_NAME
from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_results_dir
from scripts.visualisations.results.ilodo.performance_tables.performance_tables_ilodo import get_average_prediction


def _sanity_check_test_predictions(test_predictions, train_dataset):
    # Check if the train dataset is in the test predictions
    if train_dataset in test_predictions["dataset"]:
        raise ValueError(f"The training dataset ({train_dataset}) was found in the test predictions")


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
    _sanity_check_test_predictions(test_predictions, train_dataset=dataset_name)

    # Aggregate predictions
    df = get_average_prediction(test_predictions, epoch=epoch)

    # Sorting makes life easier
    df.sort_values(by=["dataset"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Add the targets and a datasets column
    ground_truths = []
    datasets = []
    for dataset in sorted(set(test_predictions["dataset"])):  # Needs to be sorted as in the df
        subject_ids = df["sub_id"][df["dataset"] == dataset]
        ground_truths.append(
            get_dataset(dataset).load_targets(target=TARGET, subject_ids=subject_ids)
        )
        datasets.extend([dataset]*len(subject_ids))
    df["ground_truth"] = numpy.concatenate(ground_truths)
    df["dataset"] = datasets

    return df[["dataset", "pred", "ground_truth"]]


TARGET = "age"

FIGSIZE = (7.4, 7.4)
FONTSIZE = 18
TITLE_FONTSIZE = FONTSIZE + 3
X_LIM = (-20, 130)
Y_LIM = X_LIM
DPI = 300
GRID_SPACING = 20
DATASET_COLORS = {"TDBrain": "#1f77b4", "MPILemon": "#ff7f0e", "HatlestadHall": "#2ca02c", "Miltiadous": "#d62728",
                  "YulinWang": "#9467bd"}


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
    models_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "performance_tables" / "models_selected"  # Models
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
                train_dataset, epoch = _read_dataset_and_epoch(line=line)
                if train_dataset not in datasets:
                    continue

                # Get the predictions and ground truth
                configuration_path = line.replace("\n", "").split(" ")[-1]
                df = _get_predictions(dataset_name=train_dataset, path=results_dir / configuration_path, epoch=epoch)

                # --------------
                # Plotting
                # --------------
                # Plot
                pyplot.figure(figsize=FIGSIZE)
                for target_dataset in datasets:
                    if target_dataset == train_dataset:
                        continue
                    indices = df["dataset"] == target_dataset
                    pyplot.plot(df["ground_truth"][indices], df["pred"][indices], ".",
                                label=PRETTY_NAME[target_dataset], color=DATASET_COLORS[target_dataset])
                pyplot.plot(X_LIM, Y_LIM, color="black")  # Plot the line y = x, which is perfect regression line (but
                # useless from a clinical perspective)

                # Plot cosmetics
                pyplot.xlim(X_LIM)
                pyplot.ylim(Y_LIM)
                pyplot.xticks(numpy.arange(*X_LIM, GRID_SPACING))
                pyplot.yticks(numpy.arange(*Y_LIM, GRID_SPACING))
                pyplot.title(f"Selection metric: {PRETTY_NAME[metric]}", fontsize=TITLE_FONTSIZE)
                pyplot.tick_params(labelsize=FONTSIZE)
                if train_dataset == datasets[-1]:
                    pyplot.xlabel("Ground truth", fontsize=FONTSIZE)
                else:
                    pyplot.tick_params(bottom=False, labelbottom=False)
                if metric == selection_metrics[0]:
                    pyplot.ylabel("Predicted", fontsize=FONTSIZE)
                else:
                    pyplot.tick_params(left=False, labelleft=False)
                pyplot.grid()
                pyplot.legend(fontsize=FONTSIZE)

                # Save the figure
                _to_path = os.path.join(
                    os.path.dirname(__file__), "plots",
                    f"predictions_lodi_{PRETTY_NAME[train_dataset].lower()}_selection_metric_{metric}"
                )
                pyplot.savefig(_to_path, dpi=DPI)


if __name__ == "__main__":
    main()
