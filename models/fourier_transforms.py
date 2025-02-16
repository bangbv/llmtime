from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, fftfreq
from models.utils import print_debug, my_print

@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x

def get_scaler(history, alpha=0.95, beta=0.3, basic=False, log_debug = False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    print(f"get_scaler history")
    # print(f"train.values history: {history}")
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            # print_debug(my_print, f"transform from :", x, log_debug)
            # print_debug(my_print, f"transform to new:", (x - min_) / q, log_debug)
            return (x - min_) / q
        def inv_transform(x):
            # print_debug(my_print, f"revert transform from :", x, log_debug)
            # print_debug(my_print, f"revert transform to new:", (x - min_) / q, log_debug)
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)

def fourier_transform(arr_signals):
    # Compute the FFT (Fast Fourier Transform)
    arr_fft_values = []
    for signal in arr_signals:
        fft_values = fft(signal)
        arr_fft_values.append(fft_values)
    return arr_fft_values


def inverse_fourier_transform_arr(arr_fft_values):
    print_debug(my_print, f"fourier_transforms:inverse_fourier_transform:arr_fft_values:", arr_fft_values, True)
    arr_reconstructed_signals = []
    for fft_values in arr_fft_values:
        print_debug(my_print, f"fourier_transforms:inverse_fourier_transform:fft_values",fft_values, True)
        print_debug(my_print, f"fourier_transforms:inverse_fourier_transform:type_fft_values:",type(fft_values), True)
        # Compute the inverse FFT
        reconstructed_signal = np.fft.ifft(fft_values)
        # Since the inverse FFT may introduce small imaginary parts due to numerical errors, take the real part
        reconstructed_signal = np.abs(reconstructed_signal.real)
        arr_reconstructed_signals.append(reconstructed_signal)
    print_debug(my_print, f"fourier_transforms:inverse_fourier_transform:arr_reconstructed_signals:", arr_reconstructed_signals, True)
    return arr_reconstructed_signals

def inverse_fourier_transform(fft_values):
    # Compute the inverse FFT
    print(f"fourier_transforms:inverse_fourier_transform:fft_values: {fft_values}")
    print(f"fourier_transforms:inverse_fourier_transform:type_fft_values: {type(fft_values)}")
    reconstructed_signal = np.fft.ifft(fft_values)
    print(f"fourier_transforms:inverse_fourier_transform:reconstructed_signal: {reconstructed_signal}")
    # Since the inverse FFT may introduce small imaginary parts due to numerical errors, take the real part
    reconstructed_signal = np.abs(reconstructed_signal.real)
    return reconstructed_signal

def fourier_transform(arr_str_signals):
    # Compute the FFT (Fast Fourier Transform)
    arr_fft_values = []
    for signal in arr_str_signals:

        fft_values = fft(signal)
        arr_fft_values.append(fft_values)
    return arr_fft_values