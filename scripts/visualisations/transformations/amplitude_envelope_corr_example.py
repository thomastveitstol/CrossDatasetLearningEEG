"""
Script for plotting the pretext task related to Amplitude Envelope Correlation
"""
from matplotlib import pyplot
import numpy


def main() -> None:
    # -----------------
    # Generating signals
    # -----------------
    # Generate abcissa parameter (time)
    num_seconds = 8
    sampling_rate = 500
    t = numpy.linspace(0, num_seconds, num=num_seconds*sampling_rate)

    # Generate pure sine wave
    freq = 2
    clean_signal = numpy.sin(2*numpy.pi*freq*t)

    # Generate envelope
    means = (1, 1.5, 2.5, 5.0, 7.2)
    stds = (.3, .5, .6, 1.0, 1.3)
    amps = (0.2, 0.5, 0.6, 0.4, 0.6)

    envelope = 1 - sum(amp*numpy.exp(-((t - mean) / std)**2) for amp, mean, std in zip(amps, means, stds))

    # Generate permutation
    permutation_mean = 4.0
    permutation_std = .4
    permutation_amp = 0.7

    permutation = 1 - permutation_amp * numpy.exp(-((t - permutation_mean)/permutation_std)**2)

    # -----------------
    # Plotting
    # -----------------
    lw = 2
    _, (ax1, ax2) = pyplot.subplots(2, 1)

    # Original
    ax1.plot(t, clean_signal*envelope, label="Original Signal, f(t)", linewidth=lw)
    ax1.plot(t, envelope, label="Original Envelope", linewidth=lw)

    # Permuted
    ax2.plot(t, clean_signal*envelope*permutation, label="Permuted Signal, f(t)*g(t)", linewidth=lw)
    ax2.plot(t, envelope*permutation, label="Permuted Envelope", linewidth=lw)
    ax2.plot(t, permutation, label="Permutation, g(t)", linewidth=lw)

    # Cosmetics
    fs = 20

    ax1.set_xlabel("t [s]", fontsize=fs)
    ax2.set_xlabel("t [s]", fontsize=fs)

    ax1.set_ylim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)

    ax1.tick_params(axis='x', labelsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)
    ax2.tick_params(axis='x', labelsize=fs)
    ax2.tick_params(axis='y', labelsize=fs)

    ax1.legend(fontsize=fs)
    ax2.legend(fontsize=fs)

    pyplot.show()


if __name__ == "__main__":
    main()
