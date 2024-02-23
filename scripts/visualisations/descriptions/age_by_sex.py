"""
Script for plotting age distribution per dataset and sex
"""
import seaborn

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    datasets = (HatlestadHall(), YulinWang(), Miltiadous(), Rockhill())

    # ----------------
    # Load data
    # ----------------
    age_distributions = dict()
    for dataset in datasets:
        age_distributions[type(dataset).__name__] = dataset.age()

    # ----------------
    # Plotting
    # ----------------
    seaborn.displot()

