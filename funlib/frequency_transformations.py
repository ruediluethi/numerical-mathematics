import numpy as np
import math
from typing import Union, Tuple

import streamlit as st

@st.cache_data
def windowed_fft(fs: float, signal: np.ndarray, return_frequencies: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the hanning-windowed FFT of a signal.
    The signal is filled with zeros to the next larger power of 2.

    Args:
        fs (float): The sampling frequency of the signal.
        signal (np.ndarray): The input signal.
        return_frequencies (bool, optional): Whether to return the frequencies. Defaults to False.

    Returns:
        np.ndarray or Tuple[np.ndarray, np.ndarray]: The windowed FFT of the signal. If `return_frequencies` is True,
        it also returns the corresponding frequencies.
    """
    # zero padd to next larger power of 2:
    sig_length = len(signal)
    fft_length = 2**(math.ceil(math.log(sig_length, 2)))
    zero_pad_length = fft_length - sig_length

    window = np.hanning(sig_length)
    sig_windowed = window*(signal-np.mean(signal))

    signal_fft = np.fft.fft(np.append(sig_windowed,np.zeros(zero_pad_length)))
    
    set_to_zero_length = int(sig_length * 0.001)
    if set_to_zero_length < 2:
        set_to_zero_length = 2
    signal_fft[:set_to_zero_length] = 0
    
    if return_frequencies:
        # adding +2 because the fft length is always even and the useful, non-zero frequencies are from 1 to (fft_length/2)+1
        # the resulting frequencies are then matching the frequencies of the fft function in gonum.
        frequencies = (np.arange(0, fft_length)*fs/(fft_length+2))
        #print(frequencies)
        return signal_fft, frequencies
    else:
        return signal_fft