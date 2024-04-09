"""
Script for plotting 2D projected channel systems
"""
from matplotlib import pyplot

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.ous_dataset import OUS
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    # Select dataset
    names = "HatlestadHall", "YulinWang", "Miltiadous", "CAUEEG"

    # Loop through all names
    for name in names:
        # Get dataset
        available_datasets = (HatlestadHall, YulinWang, Miltiadous, MPILemon, CAUEEG, OUS)
        for available_dataset in available_datasets:
            if name in (available_dataset().name, available_dataset.__name__):
                dataset = available_dataset()
                break
        else:
            raise ValueError(f"The dataset {name} was not recognised")

        # ----------------
        # Plotting
        # ----------------
        dataset.plot_2d_electrode_positions()

    pyplot.show()


if __name__ == "__main__":
    main()
