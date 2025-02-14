
import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

gaps_color = 'peru'
discrete_color = 'palevioletred'
clean_color = 'lightseagreen'

def calculate_sample_rate(time: np.ndarray) -> float:
    timediff = time[1:]-time[:-1]
    expected_timediff = np.median(timediff)
    if expected_timediff == 0: # prevent from division per zero
        expected_timediff = np.mean(timediff)
    fs = 1/round(expected_timediff,6)
    return fs

def fill_bm_with_last_value(time: np.ndarray, bmx: np.ndarray, bmy: np.ndarray):
    """
    Fill gaps in all bending moment time series data with zeros.

    Args:
        time (np.ndarray): Array of time values in seconds.
        bmx (np.ndarray): Array of Bending Moment x-axis values.
        bmy (np.ndarray): Array of Bending Moment y-axis values.
        ts (float): Sampling time in seconds.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]: Tuple containing the filled time, x-axis, y-axis arrays, and the number of inserted zeros.
    """
    fs = calculate_sample_rate(time)
    ts = 1/fs

    time_filled = np.copy(time)
    x_filled = np.copy(bmx)
    y_filled = np.copy(bmy)
    insert_idx = 1
    insert_counter = 0
    for i in range(1, len(time_filled)):
        if insert_idx >= len(time_filled):
            break
        while time[i] - time_filled[insert_idx - 1] > 1.1 * ts:
            if insert_idx >= len(time_filled):
                break
            time_filled[insert_idx] = time_filled[insert_idx - 1] + ts
            x_filled[insert_idx] = x_filled[insert_idx - 1]
            y_filled[insert_idx] = y_filled[insert_idx - 1]
            # x_filled[insert_idx] = 0
            # y_filled[insert_idx] = 0
            insert_counter += 1
            insert_idx += 1
        if insert_idx >= len(time_filled):
            break
        time_filled[insert_idx] = time[i]
        x_filled[insert_idx] = bmx[i]
        y_filled[insert_idx] = bmy[i]
        insert_idx += 1

    return time_filled, x_filled, y_filled

def discrete_fourier_transformation(f, t):
  # st.write(f.shape, t.shape)
  n = f.size

  A = np.zeros((n,n))
  B = np.zeros((n,n))

  p = t[-1] - t[0]
  # st.write(p)

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

  return frequencies[1:,]

def fast_fourier_transformation(f, duration):
  n = f.size

  # st.write(f.shape)
  # st.write(duration)

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

  return frequencies[1:,]

def von_hann_window(f, t):
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



st.title('Diskrete Fouriertransformation mit Lücken')

st.write('''
  Die diskrete Fouriertransformation ermöglicht eine Transformation ins Frequenzspektrum ohne die Notwendigkeit einer kontinuierlichen Abtastung.
  So müssen auch Datenlücken nicht zwangsläufig aufgefüllt werden, um eine Frequenzanalyse durchzuführen.
  Anhand eines Beispiels wird untersucht, wie sich unterschiedliche Strategien zur Behandlung von Datenlücken auf die Frequenzanalyse auswirken.
''')
st.page_link('pages/0_UU_Diskrete_Fouriertransformation.py', label='Hier gehts zur Theorie der diskreten Fouriertransformation', icon='🤓')


# example = st.radio('Datenbeispiel', ['mit Sensordaten', 'mit überlagerten Sinus-Testdaten'], horizontal=True)
example = 'mit überlagerten Sinus-Testdaten'

n_part = st.slider('Anzahl Datenpunkte für die Frequenzanalyse', 1, 3000, 1000, 1)

if example == 'mit überlagerten Sinus-Testdaten':

  n_peak = st.slider('Anzahl Frequenzen im Signal', 1, 10, 5, 1)

  t = np.linspace(0.0, n_part/2500, n_part)*1000

  bm = np.zeros(n_part)
  for i in range(1,n_peak+1):
      bm += np.sin( t/1000 * 2*math.pi * i *2500/2/(n_peak+1)) * np.sin(i/(n_peak+1) * math.pi)

  gap_n = st.slider('Größe der Lücke', 0, n_part, int(n_part*0.1), 1)
  gap_pos = st.slider('Position der Lücke', 0.0, 1.0, 2/3)

  gap_start_i = int((n_part-gap_n)*gap_pos)

  with_gaps_bm = np.concatenate((bm[0:gap_start_i], bm[gap_start_i+gap_n:-1]))
  with_gaps_t = np.concatenate((t[0:gap_start_i], t[gap_start_i+gap_n:-1]))


  # gaps_t, gaps_bm, gaps_bmy = fill_bm_with_last_value(with_gaps_t, with_gaps_bm, np.zeros(with_gaps_bm.size))
  gaps_t = np.copy(t)
  gaps_bm = np.copy(bm)

  gaps_bm[gap_start_i:gap_start_i+gap_n] = np.zeros(gap_n)

  # gaps_bm = np.copy(bm)

  clean_t = np.copy(t)
  clean_bm = np.copy(bm)

  # with_gaps_t = np.copy(gaps_t)
  # with_gaps_bm = np.copy(gaps_bm)


  gaps_container = st.container()
  clean_container = st.container()

else:
  raw_file_ = os.path.join('data', 'demo_sensordata.csv')
  df_ = pd.read_csv(raw_file_, skiprows=[1])
  df_.describe()

  # st.write(df_.head())

  t_with_gaps = df_['time'].to_numpy()
  t_with_gaps = t_with_gaps - t_with_gaps[0]
  bmX_with_gaps = df_['24'].to_numpy()
  bmY_with_gaps = df_['25'].to_numpy()

  start_gaps_t = st.slider('Zeitpunkt des zu Analysierenden Zeitintervals in ms', 0.0, t_with_gaps[-1]/1000, 2.5)*1000
  gap_n = st.slider('Größe der Lücke', 0, n_part, int(n_part*0.1), 1)
  gap_pos = st.slider('Position der Lücke', 0.0, 1.0, 2/3)

  gaps_container = st.container()
  start_gaps_index = np.where(t_with_gaps > start_gaps_t)[0][0]
  # start_gaps_index = 16330

  gap_start_i = start_gaps_index+int((n_part-gap_n)*gap_pos)

  t_with_gaps = np.concatenate((t_with_gaps[:gap_start_i], t_with_gaps[gap_start_i+gap_n:]))
  bmX_with_gaps = np.concatenate((bmX_with_gaps[:gap_start_i], bmX_with_gaps[gap_start_i+gap_n:]))
  bmY_with_gaps = np.concatenate((bmY_with_gaps[:gap_start_i], bmY_with_gaps[gap_start_i+gap_n:]))

  bm_with_gaps = np.sqrt(bmX_with_gaps**2 + bmY_with_gaps**2)

  

  t_, bmX_, bmY_ = fill_bm_with_last_value(t_with_gaps, bmX_with_gaps, bmY_with_gaps)
  bm_ = np.sqrt(bmX_**2 + bmY_**2)
  
  plot_container = st.container()
  # start_gaps_t = st.slider('Zeitpunkt eines Funklücken behafteten Zeitintervals in ms', 0.0, t_[-1]/1000, 2.5)*1000
  gaps_container = st.container()
  # start_gaps_index = np.where(t_ > start_gaps_t)[0][0]
  # st.write(start_gaps_index)
  start_clean_t = st.slider('Zeitpunkt eines Zeitintervals ohne Lücken in s', 0.0, t_[-1]/1000, 3.2)*1000
  clean_container = st.container()
  start_clean_index = np.where(t_ > start_clean_t)[0][0]

  gaps_t = t_[start_gaps_index:start_gaps_index+n_part]
  gaps_bm = bmX_[start_gaps_index:start_gaps_index+n_part]

  clean_t = t_[start_clean_index:start_clean_index+n_part]
  clean_bm = bmX_[start_clean_index:start_clean_index+n_part]

  gaps_indices_with_gaps = np.where((gaps_t[0] <= t_with_gaps) & (t_with_gaps <= gaps_t[-1]))[0].flatten()
  with_gaps_t = t_with_gaps[gaps_indices_with_gaps]
  with_gaps_bm = bmX_with_gaps[gaps_indices_with_gaps]




  fig, ax = plt.subplots(figsize=(8,1))
  ax.plot(t_/1000, bm_, 'lightgray', label='Signal')

  ax.plot(gaps_t/1000, gaps_bm, color=gaps_color, label='Zeitfenster mit Lücke')
  ax.plot(clean_t/1000, clean_bm, color=clean_color, label='Zeitfenster ohne Lücke')

  ax.set_xlabel('Zeit in s')
  plot_container.pyplot(fig)
  plot_container.caption('Gesamtsignal mit den ausgewählten Zeitfenstern')


gaps_t = gaps_t - gaps_t[0]
clean_t = clean_t - clean_t[0]
with_gaps_t = with_gaps_t - with_gaps_t[0]

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(gaps_t, gaps_bm, color='lightgray')
clean_bm = von_hann_window(clean_bm, clean_t)
ax.plot(clean_t, clean_bm, color=clean_color, label='ohne Lücke')
gaps_bm = von_hann_window(gaps_bm, gaps_t)
ax.plot(gaps_t, gaps_bm, color=gaps_color, label='mit Lücke')
ax.legend()

ax.set_xlabel('Zeit in ms')
with_gaps_bm = von_hann_window(with_gaps_bm, with_gaps_t)
# ax.plot(with_gaps_t, with_gaps_bm, '.', color=discrete_color, label='Zeitfenster mit Lücke')

gaps_container.pyplot(fig)
gaps_container.caption('''
  Zeitlicher Verlauf des Beispielsignals.
  Im Hintergrund grau dargestellt sind die Originaldaten.
  Für die Transformationen wurde das Signal durch eine von-Hann-Fensterfunktion weiterverarbeitet.
''')

st.write('Für die FFT des Signals mit Lücke wurden die fehlenden Datenpunkten mit Nullen aufgefüllt.')


# fig, ax = plt.subplots(figsize=(8,2))
# ax.plot(clean_t, clean_bm, color='lightgray', label='Zeitfenster ohne Lücke')
# clean_bm = von_hann_window(clean_bm, clean_t)
# ax.plot(clean_t, clean_bm, color=clean_color, label='Zeitfenster ohne Lücke')
# ax.set_xlabel('Zeit in ms')
# clean_container.pyplot(fig)
# clean_container.caption('Zeitfenster ohne Lücke mit gleicher Weiterverarbeitung durch eine von-Hann-Fensterfunktion wie das Zeitfenster mit Lücke.')

# st.write('''
#   Vergleicht man nun die Frequenzanalyse der beiden Zeitfenster, so wird deutlich, dass die Lücke im Zeitfenster zu einer Verzerrung des Frequenzspektrums führt.      
# ''')



gaps_freq = fast_fourier_transformation(gaps_bm, gaps_t[-1] - gaps_t[0])
clean_freq = fast_fourier_transformation(clean_bm, clean_t[-1] - clean_t[0])
with_gaps_freq = discrete_fourier_transformation(with_gaps_bm, with_gaps_t)


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(gaps_freq[:,0], gaps_freq[:,1]/np.max(gaps_freq[:,1]), gaps_color, label='FFT mit Lücke')
ax.plot(clean_freq[:,0], clean_freq[:,1]/np.max(clean_freq[:,1]), clean_color, label='FFT ohne Lücke')
ax.plot(with_gaps_freq[:,0], with_gaps_freq[:,1]/np.max(with_gaps_freq[:,1]), discrete_color, label='DFT mit Lücke')
ax.set_xlabel('Frequenz in Hz')
ax.set_ylabel('Normierte Amplitude')
ax.legend()
st.pyplot(fig)
st.caption('''
  Resultat der drei Transformationen im Vergleich. 
  Die Amplituden wurden jeweils durch den maximalen Wert normiert.
''')





# st.write('''
#   Wird auf dem Lücken behafteten Signal die diskrete Fouriertransformation durchgeführt, 
#   so ist das Resultat nahezu identisch mit dem Resultat der FFT auf dem Signal ohne Lücke.
# ''')

max_i = np.argmax(gaps_freq[:,1])
display_range = min(50, max_i)

fig_freq, ax_freq = plt.subplots(figsize=(8,4))
ax_freq.plot(with_gaps_freq[max_i-display_range:max_i+display_range,0], 
        with_gaps_freq[max_i-display_range:max_i+display_range,1]/np.amax(with_gaps_freq[max_i-display_range:max_i+display_range,1]), label=r'DFT mit Lücke', color=discrete_color)

ax_freq.plot(gaps_freq[max_i-display_range:max_i+display_range,0], 
        gaps_freq[max_i-display_range:max_i+display_range,1]/np.amax(gaps_freq[max_i-display_range:max_i+display_range,1]), ':', label=r'FFT mit Lücke', color=gaps_color)
ax_freq.plot(clean_freq[max_i-display_range:max_i+display_range,0], 
        clean_freq[max_i-display_range:max_i+display_range,1]/np.amax(clean_freq[max_i-display_range:max_i+display_range,1]), ':', label=r'FFT ohne Lücke', color=clean_color)

ax_freq.set_xlabel('Frequenz in Hz')
ax_freq.set_ylabel('Normierte Amplitude')
ax_freq.legend()

st.pyplot(fig_freq)
st.caption('Ausschnitt des Frequenzspektrums zur jeweils höchsten Amplitude der drei unterschiedlichen Strategien.')


st.stop()


# signal_type = st.radio('Signal', ['Bm X', 'Bm Y', 'Bm Betrag'], index=0)
# if signal_type == 'Bm Y':
#     bmX = bmY
# elif signal_type == 'Bm Betrag':
#     bmX = np.sqrt(bmX**2 + bmY**2)

# t = np.array([])
# for index, row in df.iterrows():
#     t = np.append(t, pd.Timestamp(row['time']).timestamp() * 1000)

# df_out = pd.DataFrame({'time': t, 'Bm X': bmX, 'Bm Y': bmY})
# st.write(df_out.head())

# df_out.to_csv('data/demo_sensordata2.csv', index=False)


# fig, ax = plt.subplots()
# ax.plot(np.diff(t))
# st.pyplot(fig)

# fig, ax = plt.subplots()
# ax.plot(bmX)
# st.pyplot(fig)

start_i = st.slider('start_i', 0, t.size, int(t.size/2), 1)


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

bmX_windowed = von_hann_window(bmX_parts, t_parts)

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



sig_windowed_last_value = von_hann_window(signal, t_part)
sig_windowed_last_value[end_i-start_i:start_i2-start_i] = np.ones((start_i2-end_i)).reshape((start_i2-end_i,1))*sig_windowed_last_value[end_i-start_i]
ax_raw[1].plot(t_part, sig_windowed_last_value, color='tab:orange')

frequencies = fast_fourier_transformation(sig_windowed_last_value, t_part[-1] - t_part[0])
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), label=r'letzter Wert (fft)', color='tab:orange', alpha=0.5)

# frequencies = discrete_fourier_transformation(sig_windowed_last_value, t_part)
# ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#         frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), ':', label=r'letzter Wert (diskret)', color='tab:orange')

sig_windowed_zeros = von_hann_window(signal, t_part)
sig_windowed_zeros[end_i-start_i:start_i2-start_i] = np.zeros((start_i2-end_i)).reshape((start_i2-end_i,1))
ax_raw[2].plot(t_part, sig_windowed_zeros, color='tab:green')

frequencies = fast_fourier_transformation(sig_windowed_zeros, t_part[-1] - t_part[0])
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), label=r'mit Nullen (fft)', color='tab:green', alpha=0.5)

# frequencies = discrete_fourier_transformation(sig_windowed_zeros, t_part)
# ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#         frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), ':', label=r'mit Nullen (diskret)', color='tab:green')




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
