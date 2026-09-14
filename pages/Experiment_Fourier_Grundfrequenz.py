import os
from funlib.approximation import eval_trig_approx, trig_approx
from funlib.frequency_transformations import windowed_fft
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt




raw_file = os.path.join('data', 'measurement_2022-04-13T104040812_893_0xA265.csv')
df = pd.read_csv(raw_file)


st.header("Rohdaten aus dem spike")
st.write(df.head())

# t = df["time"].to_numpy() / 1000 / 1000

@st.cache_data
def convert_time_to_seconds(df):
    t = np.array([])
    for index, row in df.iterrows():
        t = np.append(t, pd.Timestamp(row['time']).timestamp())
    return t - t[0]

t = convert_time_to_seconds(df)


fig, ax = plt.subplots()

ax.plot(t, df["Bending Moment"].to_numpy())
ax.set_xlabel("time [s]")
ax.set_ylabel("Bending Moment [Nm]")


t_start = 17200 + 0
n = 2500
n_half = n//4

t = t[t_start:t_start+n]
bm = df["Bending Moment"].to_numpy()[t_start:t_start+n]
bmx = df["Bm X"].to_numpy()[t_start:t_start+n]
bmy = df["Bm Y"].to_numpy()[t_start:t_start+n]

ax.plot(t, bm)

st.pyplot(fig)

fig, ax = plt.subplots()

ax.plot(t, bm)
ax.set_xlabel("time [s]")
ax.set_ylabel("Bending Moment [Nm]")

st.pyplot(fig)


st.header("Sinus Fit von Hand")

bm = (bm - np.mean(bm))/np.std(bm)


n_brute = 150


peak_phase = 0
peak_frequency = st.slider("peak frequency", min_value=4, max_value=n_brute, value=4, step=1)
peak_amplitude = np.std(bm)

sin_fit = peak_amplitude * np.sin(2 * np.pi * peak_frequency * t + peak_phase)

fig, ax = plt.subplots()
ax.plot(t[:n_half], bm[:n_half], label="signal")
ax.plot(t[:n_half], sin_fit[:n_half], label="sine", linewidth=2)
ax.legend()
ax.set_ylabel("amplitude")
st.pyplot(fig)

fig, ax = plt.subplots()
err = np.abs(bm[:n_half] - sin_fit[:n_half])
ax.plot(t[:n_half], err, label="error between signal and sine fit")
ax.set_xlabel("time [s]")
ax.set_ylabel("error")

st.pyplot(fig)

freq_err = np.zeros(n_brute)
for i, brute_freq in enumerate(range(n_brute)):
    peak_frequency = brute_freq
    sin_fit = peak_amplitude * np.sin(2 * np.pi * peak_frequency * t + peak_phase)
    freq_err[i] = np.sum(np.abs(bm - sin_fit))

fig, ax = plt.subplots()
ax.plot(range(n_brute)[1:], freq_err[1:])
ax.set_xlabel("frequency [Hz]")
ax.set_ylabel("error")
st.pyplot(fig)



st.header("Fast Fourier Transformation - FFT")


@st.cache_data
def calculate_sample_rate(time: np.ndarray) -> float:
    timediff = time[1:]-time[:-1]
    expected_timediff = np.median(timediff)
    if expected_timediff == 0: # prevent from division per zero
        expected_timediff = np.mean(timediff)
    fs = 1/round(expected_timediff,6)
    return fs

def poly_interp(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    calculates the coefficients for
    f_0 = a_0 + a_1*x_0 + a_2*x_0**2 + ...
    f_1 = a_0 + a_1*x_1 + a_2*x_1**2 + ...
    ...
    f_n = a_0 + a_1*x_n + a_2*x_n**2 + ... + a_n*x_n**n

    Args:
        x: A list of values.
        f: A list of function values f = f(x).

    Returns:
        The coefficients a_0, ..., a_n

    """

    # vectorized version:
    t = np.array(x).reshape(-1, 1)  # reshape f to column vector
    j = np.arange(len(f))  # create an array [0, 1, ..., n-1]
    A = t**j  # create the matrix A

    b = f.reshape(-1, 1)
    coefs = np.linalg.solve(A, b)

    return coefs

def find_peak_of_quadratic_function(f: np.ndarray, x: np.ndarray) -> float:
    """
    Finds the peak of a quadratic function.

    Args:
        x: A list of values.
        f: A list of function values f = f(x).

    Returns:
        The peak of the quadratic function: min_x f(x)

    """
    peak_coefs = poly_interp(f, x).flatten()
    return float(- peak_coefs[1] / (2*peak_coefs[2]))

def calculate_rotation_speed(freq: np.ndarray, x_fft: np.ndarray, y_fft: np.ndarray, expected_f0: float = None) -> float:
    half_len = int(len(x_fft)/2)
    estimate_fun = np.linspace(1,0,half_len)
    if expected_f0 is not None:
        # estimate_fun = norm.pdf(freq[:half_len], expected_f0, 10/1000)
        estimate_fun = 1 - np.clip(np.abs(1 - freq[:half_len] / expected_f0), 0, 1)
    else:
        estimate_fun = estimate_fun
    i_max_x = np.argmax(np.abs(x_fft[:half_len]*estimate_fun))
    i_max_y = np.argmax(np.abs(y_fft[:half_len]*estimate_fun))

    #interpolate:
    i_max_x_intp = find_peak_of_quadratic_function(np.abs(x_fft[i_max_x-1:i_max_x+2]),freq[i_max_x-1:i_max_x+2])
    i_max_y_intp = find_peak_of_quadratic_function(np.abs(y_fft[i_max_y-1:i_max_y+2]),freq[i_max_y-1:i_max_y+2])
    
    # calculate the rotation frequency from the interpolated values
    f0 = (i_max_x_intp + i_max_y_intp)/2
    # if the difference between max freq of x and y differs more than 20% from the max value of both use the min value
    if abs(i_max_x_intp - i_max_y_intp)/max(i_max_x_intp, i_max_y_intp) > 0.2:
        f0 = min(i_max_x_intp, i_max_y_intp)

    return f0, i_max_x, i_max_y


fs = calculate_sample_rate(t)
ts = 1/fs
fft, frequencies = windowed_fft(fs, bm, return_frequencies=True)

fig, ax = plt.subplots()
n_fft = frequencies.size

ax.bar(frequencies[:n_fft//2], np.abs(fft[:n_fft//2]))
st.pyplot(fig)


i_max_peak = np.argmax(np.abs(fft[:n_fft//2]))

peak_frequency = float(frequencies[i_max_peak])
peak_amplitude = float(np.abs(fft[i_max_peak]))
# peak_phase = float(np.angle(fft[i_max_peak]))
peak_phase = 0

st.write(
    {
        # "max_peak_index": i_max_peak,
        "max_peak_frequency_hz": peak_frequency,
        "max_peak_amplitude": peak_amplitude,
        "max_peak_phase_rad": peak_phase
    }
)

peak_amplitude = np.std(bm)

sin_fit = peak_amplitude * np.sin(2 * np.pi * peak_frequency * t + peak_phase)

fig, ax = plt.subplots()
ax.plot(t[:n_half], bm[:n_half], label="signal")
ax.plot(t[:n_half], sin_fit[:n_half], label="sine from max FFT peak", linewidth=2)
ax.legend()
ax.set_ylabel("amplitude")
st.pyplot(fig)

