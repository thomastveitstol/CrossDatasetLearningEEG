"""
Script for finding errors in the TDBRAIN dataset.

Manual fixes were made for sub-19703068 and sub-19703550
"""
from cdl_eeg.data.datasets.td_brain import TDBrain


def main():
    # Loop through all subjects
    for subject_id in TDBrain().get_subject_ids():
        try:
            _ = TDBrain().load_single_mne_object(subject_id=subject_id, derivatives=False, preload=False)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
