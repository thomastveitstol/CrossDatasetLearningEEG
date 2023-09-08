"""
Plot script for plotting the electrode positions of a single channel system/dataset
"""
from matplotlib import pyplot

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.mpi_lemon import MPILemon
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    # Select dataset
    name = "MPILemon"
    subject_id = None

    # Get dataset
    available_datasets = (HatlestadHall, Rockhill, YulinWang, Miltiadous, MPILemon)
    for available_dataset in available_datasets:
        if name in (available_dataset().name, available_dataset.__name__):
            dataset = available_dataset()
            break
    else:
        raise ValueError(f"The dataset {name} was not recognised")

    # ----------------
    # Plotting
    # ----------------
    dataset.plot_electrode_positions(subject_id)

    pyplot.show()


if __name__ == "__main__":
    main()
