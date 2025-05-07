import os.path
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot

from cdl_eeg.data.analysis.results_analysis import higher_is_better, PRETTY_NAME
from cdl_eeg.data.datasets.getter import get_dataset
from cdl_eeg.data.paths import get_results_dir


# -------------
# Helpful functions
# -------------
def _get_fold_from_dataset(*, dataset, path, selection_metric):
    for fold in ("Fold_0", "Fold_1", "Fold_2", "Fold_3", "Fold_4"):
        if str(path).split("/")[-2].endswith("_rerun"):
            predictions_path = f"test_history_{selection_metric}_predictions.csv"
        else:
            predictions_path = "test_history_predictions.csv"

        datasets = set(pandas.read_csv(os.path.join(path, fold, predictions_path), usecols=["dataset"])["dataset"])

        assert len(datasets) == 1

        if PRETTY_NAME[next(iter(datasets))] == dataset:
            return fold

    raise ValueError(f"Found no fold for dataset {dataset} at {path}...")


def _sanity_check_test_predictions(test_predictions, expected_dataset):
    # Check the number of datasets in the test set
    datasets = set(test_predictions["dataset"])
    if len(datasets) != 1:
        raise ValueError(f"Expected only one dataset to be present in the test set predictions, but that was not "
                         f"the case. Found {set(test_predictions['dataset'])}")

    loaded_dataset = tuple(datasets)[0]
    if PRETTY_NAME[loaded_dataset] != expected_dataset:
        raise ValueError(f"Expected dataset {expected_dataset}, but found {loaded_dataset}")


def _get_predictions(*, dataset_name, path, epoch, selection_metric):
    # Load csv file
    _prediction_columns = [f"pred{i + 1}_epoch{epoch + 1}" for i in range(NUM_PREDICTION_PER_EPOCH)]
    if str(path).split("/")[-3].endswith("_rerun"):
        predictions_path = f"test_history_{selection_metric}_predictions.csv"
    else:
        predictions_path = "test_history_predictions.csv"

    df = pandas.read_csv(os.path.join(path, predictions_path), usecols=["dataset", "sub_id"] + _prediction_columns)
    _sanity_check_test_predictions(df, expected_dataset=dataset_name)

    # Aggregate predictions
    df["avg_pred"] = df[_prediction_columns].mean(axis=1)

    # Add the targets
    df["ground_truth"] = get_dataset(_UGLY_NAME[dataset_name]).load_targets(target=TARGET, subject_ids=df["sub_id"])

    return df


# -------------
# Globals
# -------------
TARGET = "age"
NUM_PREDICTION_PER_EPOCH = 5
_UGLY_NAME = {ugly_name: pretty_name for pretty_name, ugly_name in PRETTY_NAME.items()}

# Plot cosmetics
FIGSIZE = (6.7, 6.7)
FONTSIZE = 18
TITLE_FONTSIZE = FONTSIZE + 3
X_LIM = (0, 90)
Y_LIM = (0, 90)
DPI = 300
GRID_SPACING = 10

# -------------
# Main
# -------------
def main():
    target_datasets = ("TDBRAIN", "LEMON", "SRM", "Miltiadous", "Wang")
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
        df = df[df["Source dataset"] == "Pooled"]

        # --------------
        # Loop through all the target datasets
        # --------------
        for target_dataset in target_datasets:
            filtered_df = df[df["Target dataset"] == target_dataset]

            # Get optimal model (from validation score)
            if higher_is_better(selection_metric):
                best_idx = filtered_df["Val score"].idxmax()
            else:
                best_idx = filtered_df["Val score"].idxmin()

            best_run = filtered_df["run"][best_idx]
            best_epoch = filtered_df["Epoch"][best_idx]

            # Get the predictions and ground truth
            _run_path = results_dir / best_run / "leave_one_dataset_out"
            _run_path = _run_path / _get_fold_from_dataset(dataset=target_dataset, path=_run_path,
                                                           selection_metric=selection_metric)

            predictions_df = _get_predictions(dataset_name=target_dataset, path=_run_path, epoch=best_epoch,
                                              selection_metric=selection_metric)

            # ------------
            # Plotting
            # ------------
            # Plot
            pyplot.figure(figsize=FIGSIZE)
            pyplot.plot(predictions_df["ground_truth"], predictions_df["avg_pred"], ".")
            pyplot.plot(X_LIM, Y_LIM, color="black")  # Plot line

            # Plot cosmetics
            pyplot.xlim(X_LIM)
            pyplot.ylim(Y_LIM)
            pyplot.xticks(numpy.arange(*X_LIM, GRID_SPACING))
            pyplot.yticks(numpy.arange(*Y_LIM, GRID_SPACING))
            pyplot.title(f"Selection metric: {PRETTY_NAME[selection_metric]}", fontsize=TITLE_FONTSIZE)
            pyplot.tick_params(labelsize=FONTSIZE)
            if target_dataset == target_datasets[-1]:
                pyplot.xlabel("Ground truth", fontsize=FONTSIZE)
            else:
                pyplot.tick_params(bottom=False, labelbottom=False)
            if selection_metric == selection_metrics[0]:
                pyplot.ylabel("Predicted", fontsize=FONTSIZE)
            else:
                pyplot.tick_params(left=False, labelleft=False)
            pyplot.grid()

            # Save the figure
            _to_path = os.path.join(
                os.path.dirname(__file__), "lodo_plots",
                f"predictions_lodo_{target_dataset.lower()}_selection_metric_{selection_metric}"
            )
            pyplot.savefig(_to_path, dpi=DPI)


if __name__ == "__main__":
    main()
