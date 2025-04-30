"""
Some of the rerun experiments failed, so re-running them again

LODO: {'finished_successfully': 213, 'Nothing': 1}
LODI: {'finished_successfully': 206, 'OutOfMemoryError': 1, 'Nothing': 3}
"""
import os.path
from pathlib import Path
from typing import Dict, List

import pandas

from cdl_eeg.data.paths import get_results_dir


def main():
    results_dir = Path(get_results_dir())

    reran = tuple(run for run in os.listdir(results_dir) if run.endswith("_rerun"))

    messages_lodo: Dict[str, int] = dict()
    messages_lodi: Dict[str, int] = dict()
    rerun_again: List[str] = []
    for run in reran:
        if "inverted_cv" in run:
            messages = messages_lodi
        else:
            messages = messages_lodo

        path = results_dir / run / "leave_one_dataset_out"
        for file_name in os.listdir(path):
            if file_name[-4:] == ".txt":
                message = file_name[:-4]
                break
        else:
            message = "Nothing"

        if message != "finished_successfully":
            rerun_again.append(run)

        if message in messages:
            messages[message] += 1
        else:
            messages[message] = 1

    print(messages_lodo)
    print(messages_lodi)

    # Save for re-running again
    path = Path(os.path.dirname(__file__)) / "rerun_again.csv"
    df = pandas.DataFrame({"rerun_again": rerun_again})
    df.to_csv(path, index=False)
    os.chmod(path, 0o444)


if __name__ == "__main__":
    main()
