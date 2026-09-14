import os
from funlib.approximation import eval_trig_approx, trig_approx
from funlib.frequency_transformations import windowed_fft
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

st.title("Diverse Sensordaten")

st.subheader("measurement_2022-01-18T080000000_1_0xA0F5")
df = pd.read_csv("./data/measurement_2022-01-18T080000000_1_0xA0F5.csv")
st.write(df.head())

bm = df["Bending Moment"].to_numpy()

fig, ax = plt.subplots()
ax.plot(bm)
st.pyplot(fig)

st.page_link('pages/Experiment_Bernstein.py', label='Experiment einer Approximation durch Bernstein-Polynomen', icon='🧪')




st.subheader("measurement_2022-04-13T104040812_893_0xA265")
df = pd.read_csv("./data/measurement_2022-04-13T104040812_893_0xA265.csv")
st.write(df.head())

bm = df["Bending Moment"].to_numpy()

fig, ax = plt.subplots()
ax.plot(bm)
st.pyplot(fig)

st.page_link('archive/Experiment_Phasenverschiebung.py', label='Experiment zur Phasenverschiebung zweier 90° verschobener Signale', icon='🧪')



st.subheader("demo_sensordata")
df = pd.read_csv("./data/demo_sensordata.csv")
st.write(df.head())

t = df["time"].to_numpy()
x = df["24"].to_numpy()
y = df["25"].to_numpy()
xy = np.sqrt(x**2 + y**2)

fig, ax = plt.subplots()
ax.plot(t, xy)
st.pyplot(fig)

st.page_link('pages/Experiment_DFT_mit_Luecke.py', label='Experiment einer diskreten Fourier-Transformation mit Daten-Lücke', icon='🧪')


st.subheader("demo_sensordata2")
df = pd.read_csv("./data/demo_sensordata2.csv")
st.write(df.head())

t = df["time"].to_numpy()
x = df["Bm X"].to_numpy()
y = df["Bm Y"].to_numpy()
xy = np.sqrt(x**2 + y**2)

fig, ax = plt.subplots()
ax.plot(t, xy)
st.pyplot(fig)

