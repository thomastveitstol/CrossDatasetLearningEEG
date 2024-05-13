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


def main():
    datasets = (HatlestadHall(), YulinWang(), Miltiadous(), MPILemon(), TDBrain())

    # ----------------
    # Load data
    # ----------------
    age_distributions = {"Age": [], "Dataset": []}
    for dataset in datasets:
        # Use all subjects
        subjects = dataset.get_subject_ids()

        # Loop through all subjects
        for subject in subjects:
            age_distributions["Age"].append(dataset.age(subject_ids=(subject,))[0])
            age_distributions["Dataset"].append(type(dataset).__name__)

    # ----------------
    # Plotting
    # ----------------
    ax = seaborn.kdeplot(age_distributions, hue="Dataset", x="Age", fill=True)

    # Cosmetics
    fontsize = 17
    pyplot.title("Kernel density estimates of age distribution", fontsize=fontsize + 5)
    pyplot.grid()

    pyplot.xticks(fontsize=fontsize)
    pyplot.yticks(fontsize=fontsize)
    pyplot.gca().xaxis.label.set_size(fontsize)
    pyplot.gca().yaxis.label.set_size(fontsize)

    pyplot.setp(ax.get_legend().get_texts(), fontsize=fontsize)
    pyplot.setp(ax.get_legend().get_title(), fontsize=fontsize+2)

    pyplot.show()


if __name__ == "__main__":
    main()
