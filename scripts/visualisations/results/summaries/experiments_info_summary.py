"""
Script for counting the number of successful runs for LODO and iLODO, and checking error messages

There was one unexpected errors message, a ValueError obtained 6th of June: When using mean_absolute_error of sklearn by
the Histories class, which used check_array on y_pred, assert_all_finite_element_wise raised the following:
'ValueError: Input contains NaN.' Don't know how that happened, maybe numerical instability?

The KeyboardInterrupt is expected because that's how I terminated the experiments after 2 months

The RuntimeErrors seem to be a CUDA problem, I'm getting the following when calling loss.backward:
'RuntimeError: GET was unable to find an engine to execute this computation'

When there are no files, it is because the run was killed. This occurred due to memory issues on the CPU

On the last 2 months of running, I made a mistake by exiting Pycharm. However, a folder was created, but I think it is
reasonable to not add it to the final count.

The following was printed before the re-running of some wrong experiments:
-- Number of runs --
Total number of runs: 2174
Total number of LODO runs: 1110
Total number of iLODO runs: 1064
Total number of unsuccessful LODO runs: 190
Total number of unsuccessful iLODO runs: 174
Total number of successful LODO runs: 920
Total number of successful iLODO runs: 890

-- Error messages --
Unexpected message (ValueError): age_cv_experiments_2024-06-06_062551

Messages LODO:
finished_successfully: 920
Nothing: 90
OutOfMemoryError: 74
RuntimeError: 25
ValueError: 1

Messages iLODO:
finished_successfully: 890
OutOfMemoryError: 61
RuntimeError: 16
Nothing: 97


The following was printed after the re-running of some previously wrong experiments:
-- Number of runs --
Total number of runs: 2174
Total number of LODO runs: 1110
Total number of LODI runs: 1064
Total number of unsuccessful LODO runs: 191
Total number of unsuccessful LODI runs: 178
Total number of successful LODO runs: 919
Total number of successful LODI runs: 886

-- Error messages --
Unexpected message (ValueError): age_cv_experiments_2024-06-06_062551

Messages LODO:
finished_successfully: 919
Nothing: 91
OutOfMemoryError: 74
RuntimeError: 25
ValueError: 1

Messages LODI:
finished_successfully: 886
OutOfMemoryError: 62
RuntimeError: 16
Nothing: 100
"""
import os
from typing import Dict

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.analysis.results_analysis import get_all_lodo_runs, get_all_ilodo_runs


def _count_finalised_messages(runs, results_dir, expected_messages):
    """Counts the error messages/finalised successfully"""
    message_count: Dict[str, int] = {}
    for run in runs:
        path = os.path.join(results_dir, run, "leave_one_dataset_out")

        # Loop through the files in the run folder
        for file_name in os.listdir(path):  # type: ignore
            if file_name[-4:] == ".txt":
                message = file_name[:-4]
                break
        else:
            # In my experiments, this means that the run was killed due to memory problems. Since it is killed by the
            # OS, Python can't catch it. Also, I had to restart the computer once (3rd of July) because Pycharm didn't
            # respond to anything
            message = "Nothing"

        # Increment occurrence
        if message in message_count:
            message_count[message] += 1
        else:
            message_count[message] = 1

        # Maybe notify about the run, if the message is unexpected
        if message not in expected_messages:
            print(f"Unexpected message ({message}): {run}")

    return message_count


def main():
    results_dir = get_results_dir()

    # -----------------
    # Number of runs
    # -----------------
    lodo_runs_all = get_all_lodo_runs(results_dir, successful_only=False)
    lodi_runs_all = get_all_ilodo_runs(results_dir, successful_only=False)

    # There were multiple attempts to re-run some experiments, but all those failed, so to avoid duplicates, they are
    # ignored
    lodo_runs_all = tuple(run for run in lodo_runs_all if len(run.split("--")) != 3)
    lodi_runs_all = tuple(run for run in lodi_runs_all if len(run.split("--")) != 3)

    num_tot_lodo = len(lodo_runs_all)
    num_tot_ilodo = len(lodi_runs_all)

    lodo_runs_success = get_all_lodo_runs(results_dir, successful_only=True)
    lodi_runs_success = get_all_ilodo_runs(results_dir, successful_only=True)

    # There were multiple attempts to re-run some experiments, but all those failed, so to avoid duplicates, they are
    # ignored
    lodo_runs_success = tuple(run for run in lodo_runs_success if len(run.split("--")) != 3)
    lodi_runs_success = tuple(run for run in lodi_runs_success if len(run.split("--")) != 3)

    num_successful_lodo = len(lodo_runs_success)
    num_successful_ilodo = len(lodi_runs_success)

    print(f"{' Number of runs ':-^20}")
    print(f"Total number of runs: {num_tot_lodo + num_tot_ilodo}")

    print(f"Total number of LODO runs: {num_tot_lodo}")
    print(f"Total number of LODI runs: {num_tot_ilodo}")

    print(f"Total number of unsuccessful LODO runs: {num_tot_lodo - num_successful_lodo}")
    print(f"Total number of unsuccessful LODI runs: {num_tot_ilodo - num_successful_ilodo}")

    print(f"Total number of successful LODO runs: {num_successful_lodo}")
    print(f"Total number of successful LODI runs: {num_successful_ilodo}")

    # -----------------
    # Analysing the failed experiments
    # -----------------
    print(f"\n{' Error messages ':-^20}")

    # Defining some expected messages, the others I would like to print. KeyboardInterrupt is expected because that's
    # how I terminated the experiments after 2 months
    expected_messages = {"finished_successfully", "Nothing", "OutOfMemoryError", "RuntimeError", "KeyboardInterrupt"}

    lodo_message_count = _count_finalised_messages(lodo_runs_all, results_dir=results_dir,
                                                   expected_messages=expected_messages)
    ilodo_message_count = _count_finalised_messages(lodi_runs_all, results_dir=results_dir,
                                                    expected_messages=expected_messages)

    print("\nMessages LODO:")
    for message, count in lodo_message_count.items():
        print(f"{message}: {count}")

    print("\nMessages LODI:")
    for message, count in ilodo_message_count.items():
        print(f"{message}: {count}")


if __name__ == "__main__":
    main()
