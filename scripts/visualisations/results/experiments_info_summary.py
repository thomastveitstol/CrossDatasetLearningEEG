"""
Script for counting the number of successful runs for LODO and iLODO
"""
import os
from typing import Dict

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.results_analysis import get_all_lodo_runs, get_all_ilodo_runs


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
    num_tot_lodo = len(get_all_lodo_runs(results_dir, successful_only=False))
    num_tot_ilodo = len(get_all_ilodo_runs(results_dir, successful_only=False))

    num_successful_lodo = len(get_all_lodo_runs(results_dir, successful_only=True))
    num_successful_ilodo = len(get_all_ilodo_runs(results_dir, successful_only=True))

    print(f"{' Number of runs ':-^20}")
    print(f"Total number of runs: {num_tot_lodo + num_tot_ilodo}")

    print(f"Total number of LODO runs: {num_tot_lodo}")
    print(f"Total number of iLODO runs: {num_tot_ilodo}")

    print(f"Total number of unsuccessful LODO runs: {num_tot_lodo - num_successful_lodo}")
    print(f"Total number of unsuccessful iLODO runs: {num_tot_ilodo - num_successful_ilodo}")

    print(f"Total number of successful LODO runs: {num_successful_lodo}")
    print(f"Total number of successful iLODO runs: {num_successful_ilodo}")

    # -----------------
    # Analysing the failed experiments
    # -----------------
    print(f"\n{' Error messages ':-^20}")

    lodo_runs = get_all_lodo_runs(results_dir, successful_only=False)
    ilodo_runs = get_all_ilodo_runs(results_dir, successful_only=False)

    # Defining some expected messages, the others I would like to print. KeyboardInterrupt is expected because that's
    # how I terminated the experiments after 2 months
    expected_messages = {"finished_successfully", "Nothing", "OutOfMemoryError", "RuntimeError", "KeyboardInterrupt"}

    lodo_message_count = _count_finalised_messages(lodo_runs, results_dir=results_dir,
                                                   expected_messages=expected_messages)
    ilodo_message_count = _count_finalised_messages(ilodo_runs, results_dir=results_dir,
                                                    expected_messages=expected_messages)

    print("\nMessages LODO:")
    for message, count in lodo_message_count.items():
        print(f"{message}: {count}")

    print("\nMessages iLODO:")
    for message, count in ilodo_message_count.items():
        print(f"{message}: {count}")


if __name__ == "__main__":
    main()
