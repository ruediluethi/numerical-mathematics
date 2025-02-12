
import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import random

def discrete_fourier_transformation(f, t):
  n = f.size

  A = np.zeros((n,n))
  B = np.zeros((n,n))

  p = t[-1] - t[0]

  for k in range(0,n):
    for i in range(0,n):
      A[i][k] = 2/p*math.cos(2*math.pi*k*t[i]/p)
      B[i][k] = -2/p*math.sin(2*math.pi*k*t[i]/p)

  f = f.reshape((n,1))
  a = A @ f # real part
  b = B @ f # imaginary part

  frequencies = np.zeros((math.floor(n/2),4))
  # for (let i = 3/*Math.floor(n*0.01)*/; i < n/2; i++){
  for i in range(0,math.floor(n/2)):
    frequency = i/p * 1000
    norm = math.sqrt( a[i][0]*a[i][0] + b[i][0]*b[i][0] )
    frequencies[i][0] = frequency
    frequencies[i][1] = norm
    frequencies[i][2] = a[i][0]
    frequencies[i][3] = b[i][0]

  return frequencies

def fast_fourier_transformation(f, duration):
	n = f.size

	fft = np.fft.fft(f.flatten())

	frequencies = np.zeros((math.floor(n/2),5))
    
	for i in range(0,math.floor(n/2)):
		frequency = i/duration * 1000
		a = fft[i].real
		b = fft[i].imag
		norm = math.sqrt( a*a + b*b )
		frequencies[i][0] = frequency
		frequencies[i][1] = norm
		frequencies[i][2] = a
		frequencies[i][3] = b

	return frequencies

def hamming_window(f, t):
  n = f.size
  p = t[-1] - t[0]
  
  # hamming
  # alpha = 25/46
  # beta = 1 - alpha

  # von hann
  alpha = 0.5
  beta = 0.5

  g = f * (alpha - beta * np.cos( 2 * math.pi * t / p ))
  return g.reshape((n,1))





# st.title('Diskrete Fouriertransformation mit Lücken')

raw_file = os.path.join('data', 'measurement_2022-01-18T080000000_1_0xA0F5.csv')
# raw_file = os.path.join('data', 'demo_sensordata.csv')

df = pd.read_csv(raw_file, skiprows=[1])
df.describe()

st.write(df.head())

# t = df['time'].to_numpy()
# t = (t - t[0])/1000
# bmX = df['24'].to_numpy()
# bmY = df['25'].to_numpy()

# fig, ax = plt.subplots()
# ax.plot(t, bmX)
# ax.plot(t, bmY)

bmX = df['Bm X'].to_numpy()
bmY = df['Bm Y'].to_numpy()

signal_type = st.radio('Signal', ['Bm X', 'Bm Y', 'Bm Betrag'], index=0)
if signal_type == 'Bm Y':
    bmX = bmY
elif signal_type == 'Bm Betrag':
    bmX = np.sqrt(bmX**2 + bmY**2)

t = np.array([])
for index, row in df.iterrows():
    t = np.append(t, pd.Timestamp(row['time']).timestamp() * 1000)

# fig, ax = plt.subplots()
# ax.plot(np.diff(t))
# st.pyplot(fig)

# fig, ax = plt.subplots()
# ax.plot(bmX)
# st.pyplot(fig)

st.write(t.shape)

start_i = st.slider('start_i', 0, t.size, int(t.size/2), 1)
start_i = 16330
n_part = st.slider('Anzahl Punkte für die Transformation', 1, 3000, 1250, 1)
gap = st.slider('Größe der Lücke', 0, 1000, 200, 1)
gap_pos = st.slider('Position der Lücke', 0.0, 1.0, 2/3)

end_i = start_i + math.floor((n_part-gap)*gap_pos)
start_i2 = end_i + gap
end_i2 = start_i2 + math.ceil((n_part-gap)*(1-gap_pos))

t_part = t[start_i:end_i]
t_part2 = t[start_i2:end_i2]

# t_parts = t_parts - t_parts[0]

bmX_part = bmX[start_i:end_i]
bmY_part = bmY[start_i:end_i]

bmX_part2 = bmX[start_i2:end_i2]
bmY_part2 = bmY[start_i2:end_i2]

# ax.plot(t_part, bmX_part)
# ax.plot(t_part2, bmX_part2)

# st.pyplot(fig)

t_part2 = t_part2 - t_part[0]
t_part = t_part - t_part[0]
t_parts = np.concatenate((t_part, t_part2))

bmX_parts = np.concatenate((bmX_part, bmX_part2))

bmX_windowed = hamming_window(bmX_parts, t_parts)

fig_raw, ax_raw = plt.subplots(3,1)
ax_raw[0].plot(t_part, bmX_part, color='lightgray')
ax_raw[0].plot(t_part2, bmX_part2, color='lightgray')
ax_raw[1].plot(t_part, bmX_part, color='lightgray')
ax_raw[1].plot(t_part2, bmX_part2, color='lightgray')
ax_raw[2].plot(t_part, bmX_part, color='lightgray')
ax_raw[2].plot(t_part2, bmX_part2, color='lightgray')

ax_raw[0].plot(t_part, bmX_windowed[0:end_i-start_i], color='tab:blue')
ax_raw[0].plot(t_part2, bmX_windowed[end_i-start_i-1:-1], color='tab:blue')
# ax.plot(t_part2, hanning_window(bmX_part2, t_part2), color='tab:orange')



frequencies = discrete_fourier_transformation(bmX_windowed, t_parts)

max_i = 10+np.argmax(frequencies[10:-1,1])
display_range = min(20, max_i)

fig_freq, ax_freq = plt.subplots()
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), label=r'Lücke', color='tab:blue')
# ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#         frequencies[max_i-display_range:max_i+display_range,2], label=r'Real')
# ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#         frequencies[max_i-display_range:max_i+display_range,3], label=r'Imag')
# ax.legend()
# st.pyplot(fig)



t_part = t[start_i:end_i2]
t_part = t_part - t_part[0]
bmX_part = bmX[start_i:end_i2]
bmY_part = bmY[start_i:end_i2]

signal = bmX_part

sig_length = len(signal)



sig_windowed_last_value = hamming_window(signal, t_part)
sig_windowed_last_value[end_i-start_i:start_i2-start_i] = np.ones((start_i2-end_i)).reshape((start_i2-end_i,1))*sig_windowed_last_value[end_i-start_i]
ax_raw[1].plot(t_part, sig_windowed_last_value, color='tab:orange')

frequencies = fast_fourier_transformation(sig_windowed_last_value, t_part[-1] - t_part[0])
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), label=r'letzter Wert (fft)', color='tab:orange', alpha=0.5)

frequencies = discrete_fourier_transformation(sig_windowed_last_value, t_part)
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), ':', label=r'letzter Wert (diskret)', color='tab:orange')

sig_windowed_zeros = hamming_window(signal, t_part)
sig_windowed_zeros[end_i-start_i:start_i2-start_i] = np.zeros((start_i2-end_i)).reshape((start_i2-end_i,1))
ax_raw[2].plot(t_part, sig_windowed_zeros, color='tab:green')

frequencies = fast_fourier_transformation(sig_windowed_zeros, t_part[-1] - t_part[0])
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), label=r'mit Nullen (fft)', color='tab:green', alpha=0.5)

frequencies = discrete_fourier_transformation(sig_windowed_zeros, t_part)
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), ':', label=r'mit Nullen (diskret)', color='tab:green')


ax_freq.legend()
ax_freq.set_xlabel('f in Hz')
ax_freq.set_ylabel('Amplitude normiert mit Max')
# st.pyplot(fig)

ax_raw[2].set_xlabel('t in ms')

st.pyplot(fig_raw)
st.caption('''
  Rohdaten und drei unterschiedliche Strategien wie mit der Datenlücke umgegangen wird.
  Die Rohdaten werden mit einer von-Hann-Fensterfunktion vorverarbeitet, bevor die Transformation durchgeführt wird.
''')

st.pyplot(fig_freq)
st.caption('''
  Ausschnitt aus dem Frequenzspektrum zur jeweils höchsten Amplitude der drei unterschiedlichen Strategien.
  Die Amplitude wurde durch den jeweiligen Maximalwert normiert.
''')
