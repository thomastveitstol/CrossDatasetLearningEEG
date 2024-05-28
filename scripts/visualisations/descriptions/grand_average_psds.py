"""
Script for plotting the grand average PSD for all datasets
"""
import numpy
from matplotlib import pyplot

from cdl_eeg.data.datasets.hatlestad_hall_dataset import HatlestadHall
from cdl_eeg.data.datasets.miltiadous_dataset import Miltiadous
from cdl_eeg.data.datasets.yulin_wang_dataset import YulinWang


def main():
    average_reference = True
    num_subjects_per_dataset = 150

    fmin = 0
    fmax = 60

    datasets = (HatlestadHall(), YulinWang(), Miltiadous())
    kwargs = {"HatlestadHall": {"derivatives": True, "session": "t1"},
              "YulinWang": {"derivatives": True, "visit": 1, "recording": "EC"},
              "Miltiadous": {"derivatives": True}}

    grand_averages = dict()
    for i, dataset in enumerate(datasets):
        dataset_name = type(dataset).__name__
        print(f"Computing grand average for dataset {dataset_name} ({i+1}/{len(datasets)})")

        # Use a fixed number of subjects
        subjects = dataset.get_subject_ids()[:num_subjects_per_dataset]

        # Store the excluded channels
        excluded_channels = kwargs[dataset_name].pop("excluded_channels", None)

        psds = []
        for subject in subjects:
            # --------------
            # Get and prepare the data
            # --------------
            # Get the MNE object
            raw = dataset.load_single_mne_object(subject_id=subject, **kwargs[dataset_name])

            # Maybe remove channels
            if excluded_channels is not None:
                if isinstance(excluded_channels, list):
                    raw = raw.pick(picks="eeg", exclude=excluded_channels)
                elif isinstance(excluded_channels, str):
                    raw = raw.pick(
                        picks="eeg", exclude=[ch_name for ch_name in raw.ch_names if ch_name[:3] == excluded_channels]
                    )
                else:
                    raise TypeError(f"Excluded channels type {type(excluded_channels)} not understood")

            # Maybe re-reference to average
            if average_reference:
                raw.set_eeg_reference(ref_channels="average", verbose=False)

            # --------------
            # Compute the PSD
            # --------------
            psd, _ = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False).get_data(return_freqs=True)
            psds.append(numpy.expand_dims(numpy.mean(psd, axis=0), axis=0))

        # Compute grand average
        grand_averages[dataset_name] = numpy.mean(numpy.concatenate(psds, axis=0), axis=0)

    # --------------
    # Plotting
    # --------------
    for dataset_name, psd in grand_averages.items():
        # Plot with common units (dB)
        pyplot.plot(numpy.linspace(start=fmin, stop=fmax, num=psd.shape[0]), 10*numpy.log10(psd*10**12),
                    label=dataset_name)

    # Cosmetics
    fontsize = 17
    pyplot.xlabel("Frequency [Hz]", fontsize=fontsize)
    pyplot.ylabel(r"Power [$\mu^2V^2 / Hz$] (dB)", fontsize=fontsize)
    pyplot.legend(fontsize=fontsize)
    pyplot.xticks(fontsize=fontsize)
    pyplot.yticks(fontsize=fontsize)
    pyplot.title("Grand average PSDs", fontsize=fontsize+5)

    pyplot.show()


if __name__ == "__main__":
    main()
