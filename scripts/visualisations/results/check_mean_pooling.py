"""
I discovered that the pooling method using channel averaging did not work as intended (averaged in temporal and batch
dimension too!), so this script finds the amount of affected runs'

Output (april 10, 2025):
------------------------------
Total: 920
Number of unaffected runs (LODO): 706 (76.74%)
Number of partly affected runs (LODO): 81 (8.80%)
Number of fully affected runs (LODO): 133 (14.46%)
------------------------------
Total: 890
Number of unaffected runs (LODI): 680 (76.40%)
Number of partly affected runs (LODI): 87 (9.78%)
Number of fully affected runs (LODI): 123 (13.82%)
"""
import os

from progressbar import progressbar

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.analysis.results_analysis import get_all_ilodo_runs, get_all_lodo_runs, get_config_file


def _get_affected_runs_summary(runs, results_dir):
    num_unaffected = 0
    num_partly_affected  = 0
    num_fully_affected  = 0
    for run in progressbar(runs):
        # Get config file
        config_path = os.path.join(results_dir,  run)
        config = get_config_file(results_folder=config_path, preprocessing=False)

        # If this run used interpolation, it was unaffected
        method = config["Varied Numbers of Channels"]["name"]
        if method == "Interpolation":
            num_unaffected  += 1
            continue

        assert config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling"

        design_kwargs = config["Varied Numbers of Channels"]["kwargs"]["RBPDesigns"]
        all_pooling_methods = set(design["pooling_methods"] for design in design_kwargs.values())

        if "MultiMSMean" not in all_pooling_methods:
            num_unaffected  += 1
            continue

        if len(all_pooling_methods) == 1:
            num_fully_affected  += 1
        else:
            num_partly_affected  += 1

    return num_unaffected, num_partly_affected, num_fully_affected


def _print_summary(*, num_unaffected, num_partly_affected, num_fully_affected, name):
    sum_ = num_unaffected + num_partly_affected + num_fully_affected
    num_affected = num_partly_affected + num_fully_affected
    print("-" * 30)
    print(f"Total: {sum_}\n")
    print(f"Number of unaffected runs ({name}): {num_unaffected} ({num_unaffected / sum_ * 100:.2f}%)")
    print(f"Number of affected runs ({name}): {num_affected} ({num_affected / sum_ * 100:.2f}%)")

    print(f"Number of partly affected runs ({name}): {num_partly_affected} ({num_partly_affected / sum_ * 100:.2f}%)")
    print(f"Number of fully affected runs ({name}): {num_fully_affected} ({num_fully_affected/ sum_ * 100:.2f}%)")


def _check_runs(*, runs, results_dir, name):
    num_unaffected, num_partly_affected, num_fully_affected = _get_affected_runs_summary(
        runs=runs, results_dir=results_dir)

    _print_summary(num_unaffected=num_unaffected, num_partly_affected=num_partly_affected,
                   num_fully_affected=num_fully_affected, name=name)


def main():
    results_dir = get_results_dir()

    # --------------
    # Get the results
    # --------------
    _check_runs(name="LODO", runs=get_all_lodo_runs(results_dir, successful_only=True), results_dir=results_dir)
    _check_runs(name="LODI", runs=get_all_ilodo_runs(results_dir, successful_only=True), results_dir=results_dir)

if __name__ == "__main__":
    main()
