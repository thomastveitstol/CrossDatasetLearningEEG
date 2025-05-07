import os.path
from pathlib import Path
from typing import Dict

import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import higher_is_better, PRETTY_NAME
from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_results_dir


# -------------
# Helpful functions
# -------------
def _get_fold_from_dataset(*, dataset, path):
    for fold in ("Fold_0", "Fold_1", "Fold_2", "Fold_3", "Fold_4"):
        predictions_path = "val_history_predictions.csv"
        datasets = set(pandas.read_csv(os.path.join(path, fold, predictions_path), usecols=["dataset"])["dataset"])

        assert len(datasets) == 1

        if PRETTY_NAME[next(iter(datasets))] == dataset:
            return fold

    raise ValueError(f"Found no fold for dataset {dataset} at {path}...")


def _sanity_check_test_predictions(test_predictions, expected_source_dataset):
    if expected_source_dataset in set(test_predictions["dataset"]):
        raise ValueError(f"Source dataset {expected_source_dataset} found in test predictions")


def _split_dataframe(df):
    dfs: Dict[str, pandas.DataFrame] = {}
    for dataset in set(df["dataset"]):
        dfs[dataset] = df[df["dataset"] == dataset].copy()
    return dfs


def _get_predictions(*, dataset_name, path, epoch, selection_metric):
    # Load csv file
    _prediction_columns = [f"pred{i + 1}_epoch{epoch + 1}" for i in range(NUM_PREDICTION_PER_EPOCH)]
    if str(path).split("/")[-3].endswith("_rerun"):
        predictions_path = f"test_history_{selection_metric}_predictions.csv"
    else:
        predictions_path = "test_history_predictions.csv"

    df = pandas.read_csv(os.path.join(path, predictions_path), usecols=["dataset", "sub_id"] + _prediction_columns)
    _sanity_check_test_predictions(df, expected_source_dataset=dataset_name)

    # Aggregate predictions
    df["Avg. prediction"] = df[_prediction_columns].mean(axis=1)

    # Split the dataframe to one per target dataset
    dfs: Dict[str, pandas.DataFrame] = _split_dataframe(df)

    # Add the targets
    for dataset, single_source_df in dfs.items():
        single_source_df["Ground truth"] = get_dataset(dataset).load_targets(
            target=TARGET, subject_ids=single_source_df["sub_id"])

    # Merge the dataframes
    new_df = pandas.concat(dfs.values())
    new_df["dataset"] = new_df["dataset"].replace(PRETTY_NAME)
    new_df.rename(columns={"dataset": "Dataset"}, inplace=True)
    return new_df


def _make_plot(df, *, source_dataset, datasets, selection_metrics, selection_metric):

    pyplot.figure(figsize=FIGSIZE)
    for target_dataset in datasets:
        if target_dataset == source_dataset:
            continue
        indices = df["Dataset"] == target_dataset
        pyplot.plot(df["Ground truth"][indices], df["Avg. prediction"][indices], ".",
                    label=target_dataset, color=DATASET_COLORS[target_dataset])
    pyplot.plot(X_LIM, Y_LIM, color="black")  # Plot the line y = x, which is perfect regression line (but
    # useless from a clinical perspective)

    # Plot cosmetics
    pyplot.xlim(X_LIM)
    pyplot.ylim(Y_LIM)
    pyplot.xticks(numpy.arange(*X_LIM, GRID_SPACING))
    pyplot.yticks(numpy.arange(*Y_LIM, GRID_SPACING))
    pyplot.title(f"Selection metric: {PRETTY_NAME[selection_metric]}", fontsize=TITLE_FONTSIZE)
    pyplot.tick_params(labelsize=FONTSIZE)
    if source_dataset == datasets[-1]:
        pyplot.xlabel("Ground truth", fontsize=FONTSIZE)
    else:
        pyplot.tick_params(bottom=False, labelbottom=False)
    if selection_metric == selection_metrics[0]:
        pyplot.ylabel("Avg. prediction", fontsize=FONTSIZE)
    else:
        pyplot.tick_params(left=False, labelleft=False)
    pyplot.grid()
    pyplot.legend(fontsize=FONTSIZE)

    # Save the figure
    _to_path = os.path.join(
        os.path.dirname(__file__), "lodi_plots",
        f"predictions_lodi_{source_dataset.lower()}_selection_metric_{selection_metric}"
    )
    pyplot.savefig(_to_path, dpi=DPI)


# -------------
# Globals
# -------------
TARGET = "age"
NUM_PREDICTION_PER_EPOCH = 5
_UGLY_NAME = {ugly_name: pretty_name for pretty_name, ugly_name in PRETTY_NAME.items()}

# Plot cosmetics
FIGSIZE = (7.4, 7.4)
FONTSIZE = 18
TITLE_FONTSIZE = FONTSIZE + 3
X_LIM = (-20, 130)
Y_LIM = X_LIM
DPI = 300
GRID_SPACING = 20
DATASET_COLORS = {"TDBRAIN": "#1f77b4", "LEMON": "#ff7f0e", "SRM": "#2ca02c", "Miltiadous": "#d62728",
                  "Wang": "#9467bd"}


# -------------
# Main
# -------------
def main():
    source_datasets = ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang")
    selection_metrics = ("mae", "mse", "pearson_r", "r2_score")

    results_dir = Path(get_results_dir())
    pandas.set_option("display.max_columns", 15)

    for selection_metric in selection_metrics:
        # --------------
        # Read and filter the results
        # --------------
        file_path = (Path(os.path.dirname(os.path.dirname(__file__))) /
                     f"all_test_results_selection_metric_{selection_metric}.csv")

        # Read data
        df = pandas.read_csv(file_path, usecols=["Target dataset", "Source dataset", "run", "Val score", "Epoch"])
        df = df[df["Source dataset"] != "Pooled"]

        # --------------
        # Loop through all the source datasets
        # --------------
        for source_dataset in source_datasets:
            filtered_df = df[df["Source dataset"] == source_dataset]

            # Get optimal model (from validation score)
            if higher_is_better(selection_metric):
                best_idx = filtered_df["Val score"].idxmax()
            else:
                best_idx = filtered_df["Val score"].idxmin()

            best_run = filtered_df["run"][best_idx]
            best_epoch = filtered_df["Epoch"][best_idx]

            # Get the predictions and ground truth
            _run_path = results_dir / best_run / "leave_one_dataset_out"
            _run_path = _run_path / _get_fold_from_dataset(dataset=source_dataset, path=_run_path)

            predictions_df = _get_predictions(dataset_name=source_dataset, path=_run_path, epoch=best_epoch,
                                              selection_metric=selection_metric)

            # ------------
            # Plotting
            # ------------
            # Plot
            _make_plot(predictions_df, source_dataset=source_dataset, datasets=source_datasets,
                       selection_metrics=selection_metrics, selection_metric=selection_metric)


if __name__ == "__main__":
    main()
