"""
In this script, the phase of an arbitrary EEG signal is computed and visualised

Written with ChatGPT
"""
import mne
import numpy
from matplotlib import pyplot


def main() -> None:
    # Example EEG data (create a synthetic EEG signal)
    sampling_freq = 1000  # Sampling frequency in Hz
    num_samples = 5000
    time = numpy.arange(0, num_samples) / sampling_freq
    eeg_data = numpy.sin(2 * numpy.pi * 10 * time)  # Example 10 Hz oscillation

    # Generate envelope
    means = (1, 1.5, 2.5, 5.0, 7.2)
    stds = (.3, .5, .6, 1.0, 1.3)
    amps = (0.2, 0.5, 0.6, 0.4, 0.6)

    envelope = 1 - sum(amp * numpy.exp(-((time - mean) / std) ** 2) for amp, mean, std in zip(amps, means, stds))

    eeg_data *= envelope

    # Create an MNE Raw object from the synthetic EEG data
    info = mne.create_info(ch_names=['EEG'], sfreq=sampling_freq, ch_types=['eeg'])
    raw = mne.io.RawArray([eeg_data], info)

    # Apply bandpass filtering to isolate the frequency range of interest
    raw.filter(l_freq=8, h_freq=12, fir_design='firwin')

    # Apply the Hilbert transform to extract the analytic signal
    raw.apply_hilbert(envelope=False)  # envelope=False extracts the analytic signal

    # Get the phase information
    analytic_data = raw.get_data()
    phase_data = numpy.angle(analytic_data[0])  # Phase angle of the analytic signal

    # Plot the original EEG signal and the extracted phase
    pyplot.subplot(2, 1, 1)
    pyplot.plot(time, eeg_data, label='EEG Signal')
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Amplitude')
    pyplot.legend()

    pyplot.subplot(2, 1, 2)
    pyplot.plot(time, phase_data, label='Phase')
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Phase (radians)')
    pyplot.legend()

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
