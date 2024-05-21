"""
Script for printing and creating performance metric tables
"""
import os

from cdl_eeg.data.paths import get_results_dir
from cdl_eeg.data.results_analysis import get_best_lodo_performances


def main():
    # Hyperparameters
    folder_name = "17th_of_may_holiday"
    main_metric = "pearson_r"
    balance_validation_performance = False

    # Print results
    get_best_lodo_performances(results_dir=os.path.join(get_results_dir(), folder_name), main_metric=main_metric,
                               balance_validation_performance=balance_validation_performance)


if __name__ == "__main__":
    main()
