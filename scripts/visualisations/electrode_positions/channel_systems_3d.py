"""
Plot script for plotting the electrode positions of channel systems. Just for convenience really, as it is nice to
visually check that it is as expected
"""
from typing import List

from matplotlib import pyplot

from cdl_eeg.data.datasets.dataset_base import EEGDatasetBase
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.td_brain import TDBrain
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    # Select datasets to plot
    names = "MPILemon", "HatlestadHall", "TDBrain"

    # Get datasets
    datasets: List[EEGDatasetBase] = []
    available_datasets = (HatlestadHall, YulinWang, Miltiadous, MPILemon, TDBrain)
    for name in names:
        for available_dataset in available_datasets:
            if name in (available_dataset().name, available_dataset.__name__):
                datasets.append(available_dataset())
                break
        else:
            raise ValueError(f"The dataset {name} was not recognised")

    # ----------------
    # Plotting
    # ----------------
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    for dataset in datasets:
        dataset.plot_electrode_positions(ax=ax)

    pyplot.show()


if __name__ == "__main__":
    main()
