import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# One-dimensional Fourier Transform
def create_signal():
    # Sampling parameters
    Fs = 1000  # Sampling frequency (samples per second)
    T = 1 / Fs  # Sampling interval (seconds per sample)
    N = 2000  # Number of samples
    # The linspace() function returns
    # an array of evenly spaced values within the specified interval [start, stop].
    t = np.linspace(0, N * T, N, endpoint=False)  # Time vector
    print(f"Time vector: {t}")

    # Signal parameters
    f1 = 50  # Frequency of the first sine wave (Hz)
    f2 = 120  # Frequency of the second sine wave (Hz)

    first_signal = 0.7 * np.sin(2 * np.pi * f1 * t)  # First sine wave
    second_signal = 1.0 * np.sin(2 * np.pi * f2 * t)  # Second sine wave
    # Create the signal
    combine_signal = first_signal + second_signal
    return combine_signal, N, T, first_signal, second_signal, t


def plot_signal(signal, t):
    plt.figure(figsize=(14, 6))
    plt.plot(t, signal)
    plt.title('Time-Domain Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()


def plot_two_signal(t, first_signal, second_signal):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title('First Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    # Plotting both the curves simultaneously
    plt.plot(t, first_signal, color='r', label='first signal')
    # plt.plot(t, second_signal, color='g', label='second signal')


    plt.subplot(1, 2, 2)
    plt.title('Second Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.plot(t, second_signal, color='g', label='second signal')

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    # plt.xlim(0, 200)  # Limit x-axis for better visibility
    # plt.tight_layout()
    # To load the display window
    plt.show()

def fourier_transform(signal, N, T):
    # Compute the FFT (Fast Fourier Transform)
    fft_values = fft(signal)
    freqs = fftfreq(N, T)
    return freqs, fft_values


def process_output(freqs, fft_values, N):
    # Compute the magnitude spectrum
    magnitude = np.abs(fft_values) / N  # Normalize the amplitude

    # Since the FFT output is symmetric, take only the positive half
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = magnitude[pos_mask]
    return freqs, magnitude

def plot_fourier_transform(signal, t, freqs, magnitude):
    # Plot the time-domain signal
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t, signal)
    plt.title('Time-Domain Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot the frequency-domain signal
    plt.subplot(1, 2, 2)
    plt.stem(freqs, magnitude, 'b', markerfmt=" ", basefmt="-b")
    plt.title('Frequency-Domain Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(0, 200)  # Limit x-axis for better visibility
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    signal, N, T, first_signal, second_signal, t = create_signal()
    plot_signal(signal, t)
    # plot_two_signal(t, first_signal, second_signal)
    freqs, fft_values = fourier_transform(signal, N, T)
    freqs, magnitude = process_output(freqs, fft_values, N)
    # plot_fourier_transform(signal, np.linspace(0, N * T, N, endpoint=False), freqs, magnitude)