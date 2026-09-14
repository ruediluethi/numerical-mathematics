
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

@st.cache_data
def discrete_fourier_transformation(f, t, n_clean):
	# st.write(f.shape, t.shape)
	n = f.size
	# st.write(n, n_clean)

	A = np.zeros((n,n))
	B = np.zeros((n,n))

	p = t[-1] - t[0]
	# st.write(p)

	k_values = [round(t[i] / p * n_clean) for i in range(n)]
	# k_values = np.round(np.linspace(0.0, n_clean, n))
	# k_values = [round(i / n * n_clean) for i in range(n)]
	# k_values = np.linspace(0.0, n_clean-1, n)

	for k in range(0,n):
		for i in range(0,n):
			A[i][k] = 2/p*math.cos(2*math.pi*k_values[k]*t[i]/p)
			B[i][k] = -2/p*math.sin(2*math.pi*k_values[k]*t[i]/p)
			# A[i][k] = 2/p*math.cos(2*math.pi*k*t[i]/p)
			# B[i][k] = -2/p*math.sin(2*math.pi*k*t[i]/p)
			# A[i][k] = 2/p*math.cos(2*math.pi*k*i/n)
			# B[i][k] = -2/p*math.sin(2*math.pi*k*i/n)

	# st.write(np.linalg.cond(A))

	# st.write(A)

	f = f.reshape((n,1))
	a = A @ f # real part
	b = B @ f # imaginary part

	frequencies = np.zeros((math.floor(n/2),4))
	# for (let i = 3/*Math.floor(n*0.01)*/; i < n/2; i++){
	for i in range(0,math.floor(n/2)):
		# frequency = i/p * 1000
		frequency = k_values[i] / p * 1000
		norm = math.sqrt( a[i][0]*a[i][0] + b[i][0]*b[i][0] )
		frequencies[i][0] = frequency
		frequencies[i][1] = norm
		frequencies[i][2] = a[i][0]
		frequencies[i][3] = b[i][0]

	return frequencies[1:,]

@st.cache_data
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



st.title('Diskrete Fouriertransformation mit Lücke')

st.write('''
  Die diskrete Fouriertransformation ermöglicht eine Transformation ins Frequenzspektrum ohne die Notwendigkeit einer kontinuierlichen Abtastung.
  So müssen auch Datenlücken nicht zwangsläufig aufgefüllt werden, um eine Frequenzanalyse durchzuführen.
  Anhand eines Beispiels wird untersucht, wie sich unterschiedliche Strategien zur Behandlung von Datenlücken auf die Frequenzanalyse auswirken.
''')
st.page_link('pages/0_UU_Diskrete_Fouriertransformation.py', label='Hier gehts zur Theorie der diskreten Fouriertransformation', icon='🤓')


example = st.radio('Datenbeispiel', ['mit überlagerten Sinus-Testdaten', 'mit Sensordaten'], horizontal=True)
# example = 'mit überlagerten Sinus-Testdaten'

n_part = st.slider('Anzahl Datenpunkte für die Frequenzanalyse', 1, 3000, 1000, 1)

if example == 'mit überlagerten Sinus-Testdaten':

	n_peak = st.slider('Anzahl Frequenzen im Signal', 1, 10, 5, 1)
	random_amp = st.slider('Zufällige Amplitude hinzufügen', 0.0, 2.0, 0.1, 0.01)
	
	freq = n_part/2/(n_peak+1)
	freq = st.slider('Grundfrequenz', 1, n_part, 100, 1)

	t = np.linspace(0.0, 1.0, n_part)*1000
	t_rand = np.sort(np.random.uniform(0.0, 1.0, n_part)*1000)

	bm = np.zeros(n_part)
	bm_rand = np.zeros(n_part)
	for i in range(1,n_peak+1):
		bm += np.sin( t/1000 * 2*math.pi * freq*i) * np.sin(i/(n_peak+1) * math.pi)
		bm_rand += np.sin( t_rand/1000 * 2*math.pi * freq*i) * np.sin(i/(n_peak+1) * math.pi)
      
		bm += np.random.normal(0, random_amp, n_part)
		bm_rand += np.random.normal(0, random_amp, n_part)


	gap_n = st.slider('Größe der Lücke', 0, n_part, int(n_part*0.1), 1)
	gap_pos = st.slider('Position der Lücke', 0.0, 1.0, 2/3)

	gap_start_i = int((n_part-gap_n)*gap_pos)

	if st.checkbox('Zufällig verteilte Zeitpunkte verwenden'):
		with_gaps_bm = np.concatenate((bm_rand[0:gap_start_i], bm_rand[gap_start_i+gap_n:-1]))
		with_gaps_t = np.concatenate((t_rand[0:gap_start_i], t_rand[gap_start_i+gap_n:-1]))
		# with_gaps_bm = np.copy(bm_rand)
		# with_gaps_t = np.copy(t_rand)
	else:
		# with_gaps_bm = np.concatenate((bm[0:gap_start_i], bm[gap_start_i+gap_n:-1]))
		# with_gaps_t = np.concatenate((t[0:gap_start_i], t[gap_start_i+gap_n:-1]))
		with_gaps_bm = np.copy(bm)
		with_gaps_t = np.copy(t)


	# gaps_t, gaps_bm, gaps_bmy = fill_bm_with_last_value(with_gaps_t, with_gaps_bm, np.zeros(with_gaps_bm.size))
	gaps_t = np.copy(t)
	gaps_bm = np.copy(bm)

	gaps_bm[gap_start_i:gap_start_i+gap_n] = np.zeros(gap_n)

	# gaps_bm = np.copy(bm)

	clean_t = np.copy(t)
	clean_bm = np.copy(bm)

	# with_gaps_t = np.copy(gaps_t)
	# with_gaps_bm = np.copy(gaps_bm)


	# gaps_container = st.container()
	# clean_container = st.container()

else:
  raw_file_ = os.path.join('data', 'demo_sensordata.csv')
  df_ = pd.read_csv(raw_file_, skiprows=[1])
  df_.describe()

  # st.write(df_.head())

  t_raw = df_['time'].to_numpy()
  t_raw = (t_raw - t_raw[0])/1000
  bmX_raw = df_['24'].to_numpy()
  bmY_raw = df_['25'].to_numpy()
  bm_raw = np.sqrt(bmX_raw**2 + bmY_raw**2)

  start_gaps_t = st.slider('Zeitpunkt des zu Analysierenden Zeitintervals in s', 0.0, t_raw[-1]/1000, 2.5)*1000
  plot_container = st.container()
  gap_n = st.slider('Größe der Lücke', 0, n_part, int(n_part*0.1), 1)
  gap_pos = st.slider('Position der Lücke', 0.0, 1.0, 2/3)

  # gaps_container = st.container()
  start_gaps_index = np.where(t_raw > start_gaps_t)[0][0]
  # start_gaps_index = 16330

  # gap_start_i = start_gaps_index+int((n_part-gap_n)*gap_pos)

  # t_with_gaps = np.concatenate((t_with_gaps[:gap_start_i], t_with_gaps[gap_start_i+gap_n:]))
  # bmX_with_gaps = np.concatenate((bmX_with_gaps[:gap_start_i], bmX_with_gaps[gap_start_i+gap_n:]))
  # bmY_with_gaps = np.concatenate((bmY_with_gaps[:gap_start_i], bmY_with_gaps[gap_start_i+gap_n:]))

  with_gaps_t = t_raw[start_gaps_index:start_gaps_index+n_part]
  with_gaps_bmX = bmX_raw[start_gaps_index:start_gaps_index+n_part]
  with_gaps_bmY = bmY_raw[start_gaps_index:start_gaps_index+n_part]
  # with_gaps_bm = bm_raw[start_gaps_index:start_gaps_index+n_part]
  with_gaps_bm = with_gaps_bmX

  st.write(with_gaps_t.size)

  clean_t, clean_bmX, clean_bmY = fill_bm_with_last_value(with_gaps_t, with_gaps_bmX, with_gaps_bmY)
  # clean_bm = np.sqrt(clean_bmX**2 + clean_bmY**2)
  clean_bm = clean_bmX
  st.write(clean_t.size)

  

  gaps_t = np.copy(clean_t)
  gaps_bm = np.copy(clean_bm)

  gap_start_i = int((n_part-gap_n)*gap_pos)
  gaps_bm[gap_start_i:gap_start_i+gap_n] = np.zeros(gap_n)

  gap_start_t = gaps_t[gap_start_i]
  gap_end_t = gaps_t[gap_start_i+gap_n]

  # with_gaps_gap_indices = np.where((with_gaps_t >= gap_start_t) & (with_gaps_t <= gap_end_t))[0]
  # with_gaps_t = np.delete(with_gaps_t, with_gaps_gap_indices)
  # with_gaps_bm = np.delete(with_gaps_bm, with_gaps_gap_indices)

  # with_gaps_valid_indices = with_gaps_t <= clean_t[-1]
  # with_gaps_t = with_gaps_t[with_gaps_valid_indices]
  # with_gaps_bm = with_gaps_bm[with_gaps_valid_indices]

  


  # t_, bmX_, bmY_ = fill_bm_with_last_value(t_with_gaps, bmX_with_gaps, bmY_with_gaps)
  # bm_ = np.sqrt(bmX_**2 + bmY_**2)
  
  # plot_container = st.container()
  # # start_gaps_t = st.slider('Zeitpunkt eines Funklücken behafteten Zeitintervals in ms', 0.0, t_[-1]/1000, 2.5)*1000
  # gaps_container = st.container()
  # # start_gaps_index = np.where(t_ > start_gaps_t)[0][0]
  # # st.write(start_gaps_index)
  # # start_clean_t = st.slider('Zeitpunkt eines Zeitintervals ohne Lücken in s', 0.0, t_[-1]/1000, 3.2)*1000
  # start_clean_t = start_gaps_t
  # clean_container = st.container()
  # start_clean_index = np.where(t_ > start_clean_t)[0][0]

  # gaps_t = t_[start_gaps_index:start_gaps_index+n_part]
  # gaps_bm = bm_[start_gaps_index:start_gaps_index+n_part]

  # clean_t = t_[start_clean_index:start_clean_index+n_part]
  # clean_bm = bm_[start_clean_index:start_clean_index+n_part]

  # gaps_indices_with_gaps = np.where((gaps_t[0] <= t_with_gaps) & (t_with_gaps <= gaps_t[-1]))[0].flatten()
  # with_gaps_t = t_with_gaps[gaps_indices_with_gaps]
  # with_gaps_bm = bm_with_gaps[gaps_indices_with_gaps]




  fig, ax = plt.subplots(figsize=(8,1))
  ax.plot(t_raw/1000, bmX_raw, 'lightgray', label='Rohsignal')

  ax.plot(clean_t/1000, clean_bm, color=clean_color, label='Zeitfenster')

  ax.set_xlabel('Zeit in s')
  ax.legend()
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
ax.plot(gaps_t, gaps_bm, color=gaps_color, label='mit Lücke (aufgefüllt)')

ax.set_xlabel('Zeit in ms')
with_gaps_bm = von_hann_window(with_gaps_bm, with_gaps_t)
ax.plot(with_gaps_t, with_gaps_bm, '.', color=discrete_color, label='Datenpunkte mit Lücke')

ax.legend()

st.pyplot(fig)
st.caption('''
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
with_gaps_freq = discrete_fourier_transformation(with_gaps_bm, with_gaps_t, clean_t.size)


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


st.warning('Achtung: Der folgende Teil wurde durch KI generiert')

st.write(r'''
# DFT mit Datenlücken

## Ausgangsproblem

Die klassische **FFT** (Fast Fourier Transform) ist ein effizienter Algorithmus zur Berechnung der **DFT** (Diskrete Fourier Transformation) – setzt aber gleichmässige Abtastung voraus. Bei fehlenden Datenpunkten (Lücken im Zeitsignal) bricht diese Voraussetzung.

Typische naive Lösungen haben beide Schwächen:

- **FFT mit Zero-Padding**: Lücken werden mit Nullen gefüllt → erfindet Daten, erzeugt Leakage-Artefakte
- **DFT mit $k = 0 \ldots n$** wobei $n$ = Anzahl vorhandener Punkte: Die Frequenzachse ist gestaucht, da $k$ nie die echte Nyquist-Frequenz des ursprünglichen Gitters erreicht

---

## Warum die DFT grundsätzlich mit Lücken funktioniert

Die DFT-Matrix ist definiert als:

$$A_{ik} = \frac{2}{p} \cos\!\left(2\pi k \frac{t_i}{p}\right), \quad B_{ik} = -\frac{2}{p} \sin\!\left(2\pi k \frac{t_i}{p}\right)$$

mit der Periode $p = t_{\text{end}} - t_{\text{start}}$ und den tatsächlichen Zeitstempeln $t_i$.

Das Spektrum wird durch Lösung des linearen Gleichungssystems

$$A \cdot a = f, \quad B \cdot b = f$$

berechnet – **nicht** über ein direktes Skalarprodukt. Für das Gleichungssystem ist keine Orthogonalität der Basisfunktionen nötig, solange die Matrix $A$ invertierbar ist. Analog zur Polynominterpolation mit einer Vandermonde-Matrix: die Stützstellen dürfen beliebig verteilt sein, solange das System eindeutig lösbar bleibt.

Das bedeutet: **Die DFT braucht grundsätzlich keine gleichmässige Abtastung** – die Zeitstempel $t_i$ können beliebig sein.

---

## Das Problem bei nicht-uniformen Zeitstempeln

### Mathematische Sicht

Bei gleichmässiger Abtastung gilt $t_i = i \cdot \Delta t$, und die Basisfunktionen sind **orthogonal** über die Punktmenge:

$$\sum_{i=0}^{N-1} \cos\!\left(2\pi k \frac{t_i}{p}\right) \cos\!\left(2\pi l \frac{t_i}{p}\right) = \frac{N}{2} \cdot \delta_{kl}$$

Diese Orthogonalität sorgt dafür, dass $A^T A \propto I$ – die Frequenzbins sind vollständig entkoppelt. Das Gleichungssystem vereinfacht sich zum direkten Skalarprodukt (eben der klassischen DFT-Formel), und die FFT kann es in $\mathcal{O}(N \log N)$ lösen.

Bei **zufällig verteilten** $t_i$ gilt $A^T A \not\propto I$: Die Basisfunktionen sind nicht mehr orthogonal über die konkrete Punktmenge. Das System hat zwar eine eindeutige Lösung, aber die Energie einer Frequenz "streut" in benachbarte Bins – das Spektrum wird verzerrt und verrauscht, auch wenn der Peak noch an der richtigen Stelle liegt.

### Physikalische Intuition

Eine Sinusschwingung der Frequenz $f_k$ legt über eine volle Periode $p$ genau $k$ vollständige Schwingungen zurück. Bei gleichmässiger Abtastung werden diese Schwingungen **gleichmässig beprobt**: positive und negative Halbwellen werden gleich oft und gleich dicht getroffen. Die Summe der Abtastwerte einer fremden Frequenz $f_l \neq f_k$ hebt sich exakt auf – die Basisfunktionen "sehen" einander nicht.

Bei zufällig verteilten Zeitstempeln ist diese Balance zerstört. Die Abtastpunkte häufen sich zufällig in bestimmten Phasenbereichen und lassen andere Phasenbereiche dünn besetzt. Eine Schwingung der Frequenz $f_k$ wird an ihren Nulldurchgängen kaum getroffen und an ihren Maxima unverhältnismässig oft – die Summe ist nicht mehr null. Das ist gleichbedeutend damit, dass eine Schwingung der Frequenz $f_k$ im Spektrum "aussieht" als wäre auch ein wenig Energie bei $f_l$ vorhanden, weil die Abtaststruktur die beiden nicht mehr sauber trennen kann.

Anschaulich: Gleichmässige Abtastung ist wie ein fairer Richter, der jede Phase gleich gewichtet. Zufällige Abtastung ist ein befangener Richter, der manche Phasen überbewertet und damit die Frequenzen miteinander verwechselt.

---

## Die Lösung: $k$-Mapping auf das ursprüngliche Gitter

Der entscheidende Trick: Statt $k$ gleichmässig von $0$ bis $n-1$ laufen zu lassen, werden die Frequenzindizes aus den **tatsächlichen Zeitstempeln** abgeleitet:

$$k\_\text{values}[i] = \text{round}\!\left(\frac{t_i}{p} \cdot N_\text{clean}\right)$$

wobei $N_\text{clean}$ die ursprüngliche Anzahl Datenpunkte ohne Lücken ist.

```python
k_values = [round(t[i] / p * n_clean) for i in range(n)]

for k in range(n):
    for i in range(n):
        A[i][k] = 2/p * np.cos(2*np.pi * k_values[k] * t[i] / p)
        B[i][k] = -2/p * np.sin(2*np.pi * k_values[k] * t[i] / p)
```

**Warum funktioniert das?** Das Mapping stellt sicher, dass der Term im Argument

$$k\_\text{values}[k] \cdot \frac{t_i}{p} \approx \frac{k\_\text{values}[k] \cdot k\_\text{values}[i]}{N_\text{clean}}$$

näherungsweise ganzzahlig skaliert bleibt – analog zur gleichmässigen Abtastung. Die Matrix $A$ ist damit näherungsweise orthogonal, die Frequenzbins bleiben entkoppelt, und das Spektrum ist sauber.

Die Frequenzachse bleibt korrekt:

$$f_k = \frac{k\_\text{values}[k]}{p} \cdot 1000 \quad [\text{Hz}]$$

Die Matrix bleibt **(n × n) und quadratisch** – kein Zero-Padding, kein Least-Squares, keine Regularisierung.

---

## Verhalten bei verschiedenen Lückenstrukturen

| Lückentyp | FFT (Zero-Padding) | DFT (naiv, $k=0\ldots n$) | DFT (k-Mapping) |
|---|---|---|---|
| Keine Lücken | ✅ korrekt | ✅ korrekt | ✅ korrekt |
| Eine grosse Lücke | ⚠️ Leakage | ⚠️ Stauchung | ✅ korrekt |
| Viele kleine zufällige Lücken | ⚠️ schwaches Leakage | ❌ stark gestaucht | ✅ korrekt |

---

## Fehlende Peaks im Spektrum

Im Spektrum der korrigierten DFT fehlen Frequenzbins, wo Datenpunkte fehlen. Das ist **kein Bug** sondern eine ehrliche Aussage: An diesen Frequenzen liegen keine Abtastzeitpunkte vor, das System hat dort schlicht keine Information.

Die FFT hingegen füllt diese Lücken durch die implizite Annahme gleichmässiger Abtastung – das Spektrum sieht vollständig aus, macht aber implizite Annahmen über die fehlenden Daten.

---

## Verwandter Ansatz: Lomb-Scargle

Das Lomb-Scargle-Periodogramm wurde in der Astronomie entwickelt, um Periodizitäten in unregelmässig beobachteten Lichtkurven zu finden – genau das nicht-uniforme Abtastproblem. Der Kerngedanke ist elegant: Für jede Testfrequenz $\omega$ wird ein **Phasenoffset** $\tau$ berechnet, der Sinus und Kosinus lokal orthogonalisiert:

$$\tau(\omega) = \frac{1}{2\omega} \arctan\!\left(\frac{\sum_i \sin(2\omega t_i)}{\sum_i \cos(2\omega t_i)}\right)$$

Die Leistung bei jeder Frequenz ergibt sich dann als:

$$P(\omega) = \frac{1}{2} \left[ \frac{\left(\sum_i f_i \cos(\omega(t_i - \tau))\right)^2}{\sum_i \cos^2(\omega(t_i - \tau))} + \frac{\left(\sum_i f_i \sin(\omega(t_i - \tau))\right)^2}{\sum_i \sin^2(\omega(t_i - \tau))} \right]$$

Der $\tau$-Term dreht die Basis so, dass die Projektion auf Sinus und Kosinus wieder entkoppelt ist – er korrigiert genau das Problem der fehlenden Orthogonalität über die unregelmässige Punktmenge. Das Ergebnis ist statistisch als $\chi^2$-Test interpretierbar, was eine direkte Aussage über die Signifikanz eines Peaks erlaubt.

**Einschränkung gegenüber dem $k$-Mapping-Ansatz**: Lomb-Scargle liefert nur eine **Leistungsschätzung** pro Frequenz – keine komplexen Koeffizienten $a_k$, $b_k$. Eine Rücktransformation zur Lückenfüllung ist damit nicht direkt möglich. Für reine Spektralanalyse (Welche Frequenzen sind vorhanden?) ist Lomb-Scargle eine bewährte Methode; für Rekonstruktion und Interpolation ist der hier beschriebene $k$-Mapping-Ansatz vorzuziehen.

---

## Rücktransformation und Lückenfüllung

Die berechneten Koeffizienten $a_k$, $b_k$ erlauben eine Rücktransformation an **beliebigen** Zeitpunkten:

$$x(t) = \sum_k a_k \cos\!\left(2\pi k \frac{t}{p}\right) + b_k \sin\!\left(2\pi k \frac{t}{p}\right)$$

Das ermöglicht:
- **Interpolation**: Rekonstruktion fehlender Punkte innerhalb des Analysefensters
- **Extrapolation**: Vorhersage ausserhalb des Fensters (mit den üblichen Einschränkungen)

Da die Signaleigenschaften sich zeitlich verändern können (Harmonische, Amplituden), sollte die Analyse **lokal in einem gleitenden Fenster** durchgeführt werden – analog zur STFT.
''')