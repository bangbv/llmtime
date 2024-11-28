import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def fourier_transform(signal):
    # Compute the FFT (Fast Fourier Transform)
    fft_values = fft(signal)
    return fft_values


def inverse_fourier_transform(fft_values):
    print(f'FFT values: {fft_values}')
    # Compute the inverse FFT
    reconstructed_signal = np.fft.ifft(fft_values)
    print(f'Reconstructed signal: {reconstructed_signal}')
    # Since the inverse FFT may introduce small imaginary parts due to numerical errors, take the real part
    reconstructed_signal = np.abs(reconstructed_signal.real)
    print(f'Reconstructed signal (real part): {reconstructed_signal}')
    return reconstructed_signal


def difference_ff(signal, reconstructed_signal):
    # Compute the difference between the original and reconstructed signals
    difference = np.abs(signal - reconstructed_signal)
    print(f'Difference between original and reconstructed signals: {difference}')
    return difference