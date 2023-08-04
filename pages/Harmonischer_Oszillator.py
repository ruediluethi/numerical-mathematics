import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from lib.dft import discrete_fourier_transformation, hamming_window
from lib.poly import polynomial_interpolation, calc_polyline

def simple_low_pass(f, alpha):
    f_filtered = np.zeros(f.size)
    f_filtered[0] = f[0]

    for i in range(1, f.size):
        f_filtered[i] = alpha * f[i] + (1 - alpha) * f[i-1]

    return f_filtered

def simple_high_pass(f, alpha):
    f_filtered = np.zeros(f.size)
    f_filtered[0] = f[0]

    for i in range(1, f.size):
        f_filtered[i] = alpha * f[i] + alpha * (f[i] - f[i-1])

    return f_filtered[1:f_filtered.size]



st.title('Der harmonische Oszillator')
st.header('ein Praxisbeispiel')

dir_list = os.listdir(os.path.join('data', 'oszillator'))
dir_list.reverse()

with st.sidebar:
    dir = st.selectbox('Messung', dir_list)

raw_file_acc = os.path.join('data', 'oszillator', dir, 'Accelerometer.csv')
df_acc = pd.read_csv(raw_file_acc, skiprows=[1])
df_acc.describe()

t = df_acc['seconds_elapsed'].to_numpy()
a_x = df_acc['x'].to_numpy()
a_y = df_acc['y'].to_numpy()
a_z = df_acc['z'].to_numpy()

raw_file_gyro = os.path.join('data', 'oszillator', dir, 'Gyroscope.csv')
df_gyro = pd.read_csv(raw_file_gyro, skiprows=[1])
df_gyro.describe()

v_x = df_gyro['x'].to_numpy()
v_y = df_gyro['y'].to_numpy()
v_z = df_gyro['z'].to_numpy()



raw_file_pos = os.path.join('data', 'oszillator', dir, 'Orientation.csv')
df_pos = pd.read_csv(raw_file_pos, skiprows=[1])
df_pos.describe()

pitch = df_pos['pitch'].to_numpy()
roll = df_pos['roll'].to_numpy()





disp_range = st.slider(
    'from',
    value=(0, t.size-1))

disp_size = disp_range[1]- disp_range[0]
disp_duration = (t[disp_range[1]] - t[disp_range[0]]) * 1000 # duration in ms





win_size = 50
trigger_base = np.absolute(a_z[disp_range[0]:disp_range[1]] - np.mean(a_z[disp_range[0]:disp_range[1]]))
trigger_n = math.floor(trigger_base.size/win_size)
trigger_signal = np.zeros((trigger_n, 4))

# trigger_std = np.std(trigger_signal[:,3])
# trigger_mean = np.std(trigger_signal[:,3])
trigger_std = np.std(trigger_base)
trigger_mean = np.mean(trigger_base)
trigger_threshold = trigger_mean+trigger_std

new_block_at = np.zeros(0)
movement = np.zeros(0)

peak = False
for i in range(0,trigger_n):
    win_data = trigger_base[i*win_size:i*win_size+win_size]

    max_value = np.amax(win_data)
    if (peak is False and max_value > trigger_threshold):
        new_block_at = np.append(new_block_at, [i*win_size])
        peak = True
        if (new_block_at.size > 1):
            i_start = int(disp_range[0] + new_block_at[-2])
            i_end = int(disp_range[0] + new_block_at[-1])
            pos_start = math.sqrt(pitch[i_start]*pitch[i_start] + roll[i_start]*roll[i_start])
            pos_end = math.sqrt(pitch[i_end]*pitch[i_end] + roll[i_end]*roll[i_end])
            movement = np.append(movement, [abs(pos_end - pos_start)])
    
    if (peak is True and max_value < trigger_threshold):
        peak = False

    trigger_signal[i] = [
        i*win_size,
        np.amin(win_data), 
        np.average(win_data),
        max_value]


with st.sidebar:
    block_index = st.selectbox('Block', range(1, new_block_at.size))

fig, ax = plt.subplots()
ax.set_title('Orientation')
ax.plot(pitch, label='pitch')
ax.plot(roll, label='roll')
ax.plot([disp_range[0], disp_range[0]], [-0.01, 0.01])
ax.plot([disp_range[1], disp_range[1]], [-0.01, 0.01])
ax.plot(new_block_at+disp_range[0], np.zeros(new_block_at.size), 'o', label='blocks')
ax.plot([new_block_at[block_index-1]+disp_range[0], new_block_at[block_index]+disp_range[0]], 
        [0, 0], label='current block')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title('Accelerometer')
ax.plot(trigger_base, label='z')
# ax.plot(trigger_signal[:,0], trigger_signal[:,1], label='min')
# ax.plot(trigger_signal[:,0], trigger_signal[:,2], label='avg')
ax.plot(trigger_signal[:,0], trigger_signal[:,3], label='max')
ax.plot([0, trigger_base.size], [trigger_mean, trigger_mean], label='max mean')
ax.plot([0, trigger_base.size], [trigger_threshold, trigger_threshold], label='max std dev')
ax.plot(new_block_at, np.ones(new_block_at.size)*trigger_mean, 'o', label='blocks')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title('Movement')
ax.plot(movement)
ax.legend()
st.pyplot(fig)


st.subheader('Detail')


# for i in range(1, new_block_at.size):
i_start = int(disp_range[0] + new_block_at[block_index-1])
i_end = int(disp_range[0] + new_block_at[block_index])

fig, ax = plt.subplots()
ax.set_title('Accelerometer')
ax.plot(t[i_start:i_end], a_z[i_start:i_end], label='z')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title('Gyroscope')
ax.plot(v_x[i_start:i_end], label='x')
ax.plot(v_y[i_start:i_end], label='y')
ax.legend()
st.pyplot(fig)

frequencies_v_x = discrete_fourier_transformation(
    hamming_window(v_x[i_start:i_end]), 
    disp_duration)
frequencies_v_y = discrete_fourier_transformation(
    hamming_window(v_y[i_start:i_end]), 
    disp_duration)



filtered_pitch = pitch[i_start:i_end]
filtered_roll = roll[i_start:i_end]

filtered_pitch = filtered_pitch - np.mean(filtered_pitch)
filtered_roll = filtered_roll - np.mean(filtered_roll)

filtered_pitch = filtered_pitch - filtered_pitch[0]
filtered_roll = filtered_roll - filtered_roll[0]

fig, ax = plt.subplots()
ax.set_title('Orientation')
# ax.plot(pitch[i_start:i_end], label='pitch')
ax.plot(filtered_pitch, label='pitch filtered')
# ax.plot(roll[i_start:i_end], label='roll')
ax.plot(filtered_roll, label='roll filtered')
# ax.legend()
st.pyplot(fig)

frequencies_pitch = discrete_fourier_transformation(
    hamming_window(filtered_pitch), 
    disp_duration)
frequencies_roll = discrete_fourier_transformation(
    hamming_window(filtered_roll), 
    disp_duration)

freq_range = st.slider('Frequencies display range', value=(0, frequencies_v_x[:,0].size))

fig, ax = plt.subplots()
ax.set_title('Frequencies')
ax.plot(frequencies_v_x[freq_range[0]:freq_range[1],0], 
        frequencies_v_x[freq_range[0]:freq_range[1],1] / np.amax(frequencies_v_x[:,1]), '-', label='x')
ax.plot(frequencies_v_y[freq_range[0]:freq_range[1],0], 
        frequencies_v_y[freq_range[0]:freq_range[1],1] / np.amax(frequencies_v_y[:,1]), '-', label='y')
ax.plot(frequencies_pitch[freq_range[0]:freq_range[1],0], 
        frequencies_pitch[freq_range[0]:freq_range[1],1] / np.amax(frequencies_pitch[:,1]), ':', label='pitch')
ax.plot(frequencies_roll[freq_range[0]:freq_range[1],0], 
        frequencies_roll[freq_range[0]:freq_range[1],1] / np.amax(frequencies_roll[:,1]), ':', label='roll')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title('Bode-Plot')
ax.plot(frequencies_v_x[freq_range[0]:freq_range[1],0], 
        frequencies_v_x[freq_range[0]:freq_range[1],2], label='real')
ax.plot(frequencies_v_x[freq_range[0]:freq_range[1],0], 
        frequencies_v_x[freq_range[0]:freq_range[1],3], label='imag')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title('Bode-Plot')
ax.plot(frequencies_v_x[freq_range[0]:freq_range[1],0], 
        frequencies_v_x[freq_range[0]:freq_range[1],4])
ax.legend()
st.pyplot(fig)

v_y_i_max = np.argmax(v_y[i_start:i_end]) + i_start

freq_v_y_i_max = np.argmax(frequencies_v_y[:,1])
freq_v_y_max = frequencies_v_x[freq_v_y_i_max,0]
st.write(freq_v_y_max)

t_part = t[v_y_i_max:i_end] - t[v_y_i_max]

sin_wave = np.zeros(t_part.size)
for i in range(0, t_part.size):
    sin_wave[i] = math.cos(2*math.pi * t_part[i] / freq_v_y_max)*0.1


fig, ax = plt.subplots()
ax.set_title('Gyroscope')
ax.plot(t_part, v_y[v_y_i_max:i_end], label='y')
ax.plot(t_part, sin_wave, label='sin')
ax.legend()
st.pyplot(fig)