"""
Script for downloading the dataset provided at http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html
"""
from cdl_eeg.data.datasets.mpi_lemon import MPILemon


def main():
    MPILemon().download()


if __name__ == "__main__":
    main()
