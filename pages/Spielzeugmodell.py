import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


dir_list = os.listdir(os.path.join('data', 'spielzeugmodell'))
dir_list.reverse()

with st.sidebar:
    dir = st.selectbox('Messung', dir_list)

raw_file_acc = os.path.join('data', 'spielzeugmodell', dir, 'Accelerometer.csv')
df_acc = pd.read_csv(raw_file_acc, skiprows=[1])
df_acc.describe()

t = df_acc['seconds_elapsed'].to_numpy()
a_x = df_acc['x'].to_numpy()
a_y = df_acc['y'].to_numpy()
a_z = df_acc['z'].to_numpy()

raw_file_gyro = os.path.join('data', 'spielzeugmodell', dir, 'Gyroscope.csv')
df_gyro = pd.read_csv(raw_file_gyro, skiprows=[1])
df_gyro.describe()

v_x = df_gyro['x'].to_numpy()
v_y = df_gyro['y'].to_numpy()
v_z = df_gyro['z'].to_numpy()



raw_file_pos = os.path.join('data', 'spielzeugmodell', dir, 'Orientation.csv')
df_pos = pd.read_csv(raw_file_pos, skiprows=[1])
df_pos.describe()

pitch = df_pos['pitch'].to_numpy()
roll = df_pos['roll'].to_numpy()





disp_range = st.slider(
    'from',
    value=(0, t.size-1))

disp_size = disp_range[1]- disp_range[0]
disp_duration = (t[disp_range[1]] - t[disp_range[0]]) * 1000 # duration in ms



fig, ax = plt.subplots()
ax.set_title('Accelerometer')
ax.plot(a_x[disp_range[0]:disp_range[1]], label='x')
ax.plot(a_y[disp_range[0]:disp_range[1]], label='y')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title('Gyroscope')
ax.plot(v_x[disp_range[0]:disp_range[1]], label='x')
ax.plot(v_y[disp_range[0]:disp_range[1]], label='y')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title('Orientation')
ax.plot(pitch[disp_range[0]:disp_range[1]], label='pitch')
ax.plot(roll[disp_range[0]:disp_range[1]], label='roll')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(v_x[disp_range[0]:disp_range[1]], v_y[disp_range[0]:disp_range[1]], '.')
ax.legend()
st.pyplot(fig)