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
# st.write(df.head())

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

st.header("Signal ist Summe aus mehreren Frequenzen")

@st.cache_data
def rotation_speed(bmx, bmy):
    x_fft, frequencies = windowed_fft(fs, bmx, return_frequencies=True)
    y_fft = windowed_fft(fs, bmy)


    # fig, ax = plt.subplots(2,1)
    # ax[0].bar(frequencies[:n_fft//2], np.abs(x_fft[:n_fft//2]))
    # ax[1].bar(frequencies[:n_fft//2], np.abs(y_fft[:n_fft//2]))
    # st.pyplot(fig)

    # calculate the rotation speed
    fr, i_max_x, i_max_y = calculate_rotation_speed(frequencies, x_fft, y_fft)
    # tr = 1/fr
    return fr

fr_calc = rotation_speed(bmx, bmy)
# st.write(fr_calc)

fr = st.slider("Umdrehungsfrequenz", min_value=fr_calc-1, max_value=fr_calc+1, value=26.74, step=0.01, format="%.2f")


tr = 1/fr

t_rad = (t+np.pi*0.1) / tr *2*np.pi

approx_order = st.slider("approximation order", min_value=0, max_value=30, value=0)
coefs_bm = trig_approx(bm, t_rad, approx_order)

t_eq = np.linspace(0, 2*np.pi, 1000)
bm_approx = eval_trig_approx(t_eq, coefs_bm)

def rotation_time_projection(t):
    return t / (2*np.pi) * tr

fig, ax = plt.subplots()
ax.plot(rotation_time_projection(t_rad % (2*np.pi)), bm, '.')
for i in range(1, approx_order+1):
    ax.plot(rotation_time_projection(t_eq), 0.5*np.real(coefs_bm[0]) + np.real(coefs_bm[i])*np.cos(i*t_eq) + np.imag(coefs_bm[i])*np.sin(i*t_eq), label=f"order {i}")

ax.plot(rotation_time_projection(t_eq), bm_approx, color="white", linewidth=4)
ax.plot(rotation_time_projection(t_eq), bm_approx, label="Summe", color="black", linewidth=2)
ax.legend()
ax.set_xlabel("Zeit für eine Umdrehung [s]")
ax.set_ylabel("Biegemoment-Betrag [Nm]")
st.pyplot(fig)



coefs_x = trig_approx(bmx, t_rad, approx_order)
coefs_y = trig_approx(bmy, t_rad, approx_order)

bmx_approx = eval_trig_approx(t_eq, coefs_x)
bmy_approx = eval_trig_approx(t_eq, coefs_y)

fig, ax = plt.subplots(2,1)
ax[0].plot(rotation_time_projection(t_rad % (2*np.pi)), bmx, '.')
ax[1].plot(rotation_time_projection(t_rad % (2*np.pi)), bmy, '.')
ax[0].plot(rotation_time_projection(t_eq), bmx_approx, color="white", linewidth=4)
ax[1].plot(rotation_time_projection(t_eq), bmy_approx, color="white", linewidth=4)
ax[0].plot(rotation_time_projection(t_eq), bmx_approx, label="Summe", color="black", linewidth=2)
ax[1].plot(rotation_time_projection(t_eq), bmy_approx, label="Summe", color="black", linewidth=2)
ax[0].set_ylabel("X-Richtung")
ax[1].set_ylabel("Y-Richtung")
st.pyplot(fig)


fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.plot(bmx, bmy, '.')
ax.plot(bmx_approx, bmy_approx, label="Summe", color="white", linewidth=4)
ax.plot(bmx_approx, bmy_approx, label="Summe", color="black", linewidth=2)
ax.set_xlabel("X-Richtung")
ax.set_ylabel("Y-Richtung")
st.pyplot(fig)