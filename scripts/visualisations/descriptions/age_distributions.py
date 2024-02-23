"""
Script for plotting age distribution per dataset and sex
"""
import numpy
import seaborn
from matplotlib import pyplot

from cdl_eeg.data.datasets.cau_eeg_dataset import CAUEEG
from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.rockhill_dataset import Rockhill
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def _sex_int_to_str(sex):
    if sex == 0:
        return "Male"
    elif sex == 1:
        return "Females"
    else:
        raise ValueError(f"Unexpected sex value: {sex}")


def main():
    datasets = (HatlestadHall(), YulinWang(), Miltiadous(), Rockhill(), CAUEEG())

    # ----------------
    # Load data
    # ----------------
    age_distributions = {"Age": [], "Dataset": [], "Sex": []}
    for dataset in datasets:
        # Use all subjects
        subjects = dataset.get_subject_ids()

        # Loop through all subjects
        for subject in subjects:
            age_distributions["Age"].append(dataset.age(subject_ids=(subject,))[0])
            age_distributions["Dataset"].append(type(dataset).__name__)
            try:
                age_distributions["Sex"].append(_sex_int_to_str(dataset.sex((subject,))[0]))
            except AttributeError:
                age_distributions["Sex"].append(numpy.nan)

    # ----------------
    # Plotting
    # ----------------
    seaborn.histplot(age_distributions, hue="Dataset", x="Age", kde=True)

    pyplot.show()


if __name__ == "__main__":
    main()
