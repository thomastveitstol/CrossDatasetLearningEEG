import os
from pathlib import Path

import pandas

from cdl_eeg.data.analysis.results_analysis import get_all_lodo_runs, get_all_ilodo_runs, get_config_file
from cdl_eeg.data.paths import get_results_dir


def _get_2024_only(runs, *, prefix: str):
    """Get the runs from 2024 only"""
    acceptable_runs = []
    prefix_length = len(prefix)
    for run in runs:
        assert run.startswith(prefix)
        if run[prefix_length:(prefix_length + 4)] == "2024":
            acceptable_runs.append(run)
    return tuple(acceptable_runs)


def _get_affected_runs(runs, *, results_dir):
    """Get all affected runs. The input should all be from 2024, but it is not checked here"""
    affected_runs = []
    for run in runs:
        # Get config file
        config_path = os.path.join(results_dir, run)
        config = get_config_file(results_folder=config_path, preprocessing=False)

        # If this run used interpolation, it was unaffected
        method = config["Varied Numbers of Channels"]["name"]
        if method == "Interpolation":
            continue

        assert config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling"

        design_kwargs = config["Varied Numbers of Channels"]["kwargs"]["RBPDesigns"]
        all_pooling_methods = set(design["pooling_methods"] for design in design_kwargs.values())

        if "MultiMSMean" in all_pooling_methods:
            affected_runs.append(run)

    return tuple(affected_runs)


def main():
    results_dir = get_results_dir()

    # Get all runs that were part of the initial experiments (indicated by being from 2024)
    lodi_runs = _get_2024_only(get_all_ilodo_runs(results_dir, successful_only=True),
                               prefix="age_inverted_cv_experiments_")
    lodo_runs = _get_2024_only(get_all_lodo_runs(results_dir, successful_only=True),
                               prefix="age_cv_experiments_")

    # Get the affected ones
    affected_lodi = _get_affected_runs(runs=lodi_runs, results_dir=results_dir)
    affected_lodo = _get_affected_runs(runs=lodo_runs, results_dir=results_dir)

    # Save as a .csv file
    df = pandas.DataFrame({"affected": affected_lodi + affected_lodo})
    assert len(df["affected"]) == len(set(df["affected"]))
    path = Path(os.path.dirname(__file__)) / "affected_runs.csv"
    df.to_csv(path, index=False)
    os.chmod(path, 0o444)


if __name__ == "__main__":
    main()
