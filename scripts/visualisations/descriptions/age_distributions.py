"""
Script for plotting age distribution of the datasets
"""
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.td_brain import TDBrain
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang
from cdl_eeg.data.results_analysis import PRETTY_NAME


def main():
    datasets = (TDBrain(), MPILemon(), HatlestadHall(), Miltiadous(), YulinWang())

    # ----------------
    # Load data
    # ----------------
    age_distributions = {"Age": [], "Dataset": []}
    for dataset in datasets:
        # Use all subjects
        subjects = dataset.get_subject_ids()

        # Add ages and dataset
        age_distributions["Age"].extend(dataset.age(subject_ids=subjects))
        age_distributions["Dataset"].extend([PRETTY_NAME[type(dataset).__name__]] * len(subjects))

    # ----------------
    # Plotting
    # ----------------
    pyplot.figure(figsize=(16, 5))
    ax = seaborn.kdeplot(age_distributions, hue="Dataset", x="Age", fill=True)

    # Cosmetics
    fontsize = 17
    pyplot.title("Age distributions", fontsize=fontsize + 5)
    pyplot.grid()

    pyplot.xticks(fontsize=fontsize)
    pyplot.yticks(fontsize=fontsize)
    pyplot.gca().xaxis.label.set_size(fontsize)
    pyplot.gca().yaxis.label.set_size(fontsize)

    pyplot.setp(ax.get_legend().get_texts(), fontsize=fontsize)
    pyplot.setp(ax.get_legend().get_title(), fontsize=fontsize+2)

    pyplot.tight_layout()

    pyplot.show()


if __name__ == "__main__":
    main()
