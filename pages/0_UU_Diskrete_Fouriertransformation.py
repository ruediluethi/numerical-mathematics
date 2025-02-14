import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import random

st.title('Diskrete Fouriertransformation')


st.header('Fourier-Reihen')

st.subheader('Definition der trigonometrischen Reihe')

st.write(r'''
  Eine trigonometrische Reihe mit Periode $p > 0$ ist eine Funktion $s$ der Form
''')

st.latex(r'''
  s(x) = \frac{a_0}{2} +
	\sum_{k=1}^\infty a_k \cos\left( \frac{2\pi k x}{p} \right) +
	\sum_{k=1}^\infty b_k \sin\left( \frac{2\pi k x}{p} \right)
	= \sum_{k=-\infty}^\infty c_k e^{i \frac{2\pi k x}{p} }
''')

st.write(r'''
  Mit dem Skalarprodukt $\left\langle f,g \right\rangle = \int_0^p f(x) \overline{g(x)} ~dx$
  bildet die trigonometrische Reihe eine Orthonormalbasis für den Funktionenraum $\mathbb{C}^\infty$.
  Denn für $k \neq l$ gilt
''')

st.latex(r'''
  \left\langle
	e^{i \frac{2\pi l x}{p}},
	e^{i \frac{2\pi k x}{p}}
	\right\rangle =
	\int_0^p e^{i \frac{2\pi l x}{p}}
	\overline{ e^{i \frac{2\pi k x}{p}} } ~dx
	= \int_0^p e^{i \frac{2\pi (l-k) x}{p}} ~dx\\
	= \int_0^p \cos\left( \frac{2\pi (l-k) x}{p} \right) +
	i \sin\left( \frac{2\pi (l-k) x}{p} \right) ~dx\\
	= \left( \sin\left( 2\pi(l-k) \right) - \sin\left( 0 \right) \right) +
	i\left( -\cos\left( 2\pi(l-k) \right) - (-1)\cos\left( 0 \right) \right) \\
	= \left( 0 - 0 \right) + i\left(-1 + 1\right) =  0
''')


st.subheader('Diskrete Fouriertransformation')

st.write(r'''
  Sei $f(t)$ das zu messende Signal, welches zu den diskreten Zeitpunkten $t_0, ..., t_n$ die Messwerte $f_0 = f(t_0), ..., f_n = f(t_n)$ annimmt.
  Sei $f$ mit $f_0, ..., f_n$ ein Vektor mit den diskrete Messswerten zu den Zeitpunkten $t_0, ..., t_n$, 
  welcher nun in den Funktionsraum mit Basis $s_k = \overline{e^{i \frac{2\pi k}{p}t}}$ transformiert wird.
         
  Es werden also die Koeffizienten $c_k$ gesucht, so dass $f = \sum_{k=0}^n c_k s_k$ gilt.
''')

st.latex(r'''
  c_k
  = \left\langle f, s_k \right\rangle
  = \frac{1}{p} \int_0^p f(t) \cdot \overline{e^{i \frac{2\pi k}{p}t}} ~dt
  \stackrel{\substack{\textrm{Diskret-}\\\textrm{isieren}}}{=}
  \frac{1}{t_n - t_0} \sum_{l=0}^n f_l \cdot \overline{e^{i \frac{2\pi k }{t_n - t_0}t_l}}
''')

st.write(r'''
  aus $c_k \in \mathbb{C}$ lassen sich $a_k, b_k \in \R$ bestimmen
''')

st.latex(r'''
  \begin{align*}
    a_k = 2~\textrm{Re}(c_k) &= \frac{2}{p} \int_0^p f(t) \cos\left( \frac{2\pi k t}{p} \right) ~dt
    \stackrel{\substack{\textrm{Diskret-}\\\textrm{isieren}}}{=}
    \frac{2}{t_n - t_0} \sum_{l=0}^n f_l \cdot \cos\left( \frac{2\pi k x_l}{t_n - t_0} \right)\\
    b_k = -2~\textrm{Im}(c_k) &= -\frac{2}{p} \int_0^p f(x) \sin\left( \frac{2\pi k x}{p} \right) ~dx
  \end{align*}
''')

st.header('Grundlagen')

st.subheader('Cauchy Folge')

st.write(r'''
  Eine Folge $\left( a_i \right)_{i \in \mathbb{N} }$ heißt **Cauchy-Folge** wenn
  für jedes $\varepsilon$ ein Index $N$ exisitert, ab welchem die Differenz der
  Folgeglieder für $m,n > N$ kleiner als $\varepsilon$ sind:
''')

st.latex(r'''
  \forall \varepsilon > 0 \quad
  \exists N \in \mathbb{N} \quad
  \forall m,n > N: \quad
  \left\vert a_m - a_n \right\vert < \varepsilon
''')

st.subheader('Vektorraum')

st.write(r'''
  Vektorraum $V$ wird definiert durch die Vektoradditionaxiome
  - V1 (Assoziativ): $u + (v + w) = (u + v) + w$
  - V2 (neutrales Element): $0 \in V$ mit $v + 0 = 0 + v = v$
  - V3 (inverses Element): $\forall v \in V \exists -v \in V$ mit $v + (-v) = (-v) + v = 0$
  - V4 (Kommutativ): $u + v = v + u$

  und die Skalarmultiplikationsaxiome (S1 ... S4)
  - S1 (Einheitselement): $1 \cdot v = v$
  - S2 (Distributativ): $\alpha (u + v) = \alpha u + \alpha v$
  - S3 (Distributativ): $( \alpha + \beta ) v = \alpha v + \beta v$
  - S4 (Assoziativ): $( \alpha \cdot \beta ) \cdot v = \alpha \cdot ( \beta \cdot v )$
''')

st.subheader('Norm')

st.write(r'*Axiome...*')


st.subheader('Banachraum')

st.write(r'''
  Sei $V$ ein Vektorraum und $\left\Vert\cdot\right\Vert$ eine Norm.
  Konvergiert in $V$ jede Cauchy-Folge $\left(f_n\right)_{n\in\N}$,
  so ist $V$ vollständig und heißt **Banachraum**
  $\left( V, \left\Vert\cdot\right\Vert \right)$.
''')

st.latex(r'''
	\forall \varepsilon > 0 \quad
	\exists N \in \mathbb{N} \quad
	\forall m,n > N: \quad
	\left\Vert f_k - f_l \right\Vert < \varepsilon
''')

st.subheader('Skalarprodukt')

st.write(r'*Axiome...*')

st.write(r'''
  Orthogonale Projektion $b_a$ eines Vektors $b$ auf einen anderen Vektor $a$
''')

st.latex(r'''
  b_a = \underbrace{\left\langle b , a \right\rangle}_{Skalarprodukt} \cdot
  \underbrace{\frac{a}{\Vert a \Vert}}_{\substack{Normalisierter\\Vektor}}
''')

st.subheader('Hilbertraum')

st.write(r'''
  Wird eine Norm $\left\Vert \cdot \right\Vert$ durch
  ein Skalarprodukt $\left\langle \cdot,\cdot \right\rangle$ induziert,
  so ist der dazugehörige Banachraum
  $\left( V, \left\langle \cdot,\cdot \right\rangle \right)$
  ein **Hilbertraum**.
''')



st.subheader('Basis und lineare Unabhängigkeit')

st.write(r'''
  Sei $b_1, ..., b_n$ eine Basis des Unterraums $U \subset V$ mit Dimension $n$, so gilt
''')

st.latex(r'''
  \forall u \in U \quad
  \exists u_1, ..., u_n \in \mathbb{R}: \quad
  u_1 \cdot b_1 + ... + u_n \cdot b_n = u
''')

st.write(r'''
  Sind die Basis-Vektoren normiert
  $\left\Vert b_1 \right\Vert , ..., \left\Vert b_n \right\Vert = 1$
  und ist $U$ ein Hilbertraum, so gilt
''')

st.latex(r'''
  u_i = \left\langle u, b_i \right\rangle
''')

st.write(r'''
  $u_i$ ist die Länge der Projektion von $u$ auf $b_i$
''')


st.subheader('Normalengleichungen')

st.write(r'''
  Sei $V$ ein Hilbertraum und $U \subset V$ ein Unterraum von $V$.
  Gesucht sei ein Proximum $\overline{u} \in U$ mit
  minimaler Distanz zu einem $v \in V$.
''')

st.latex(r'''
  \left\Vert v - \overline{u} \right\Vert = \min_{u\in U} \left\Vert v - u \right\Vert
''')



st.write(r'''
  **Beweis**

  (i) Sei $\overline{u}$ ein Proximum, so hat die Funktion
  $F(\varepsilon) = \left\Vert v - \overline{u} \right\Vert$
  für alle beliebigen $u \in U$
  per Definition ein Minimum bei $\varepsilon = 0$
''')

st.latex(r'''
  F(\varepsilon)
  = \left\Vert v - \overline{u} \right\Vert
  = \left\langle v - \left( \overline{u} + \varepsilon u \right), v - \left( \overline{u} + \varepsilon u \right) \right\rangle \\
  = \left\Vert v \right\Vert^2 - 2\left( \overline{u} + \varepsilon u \right)^\top v +
  \underbrace{\left\Vert \overline{u} + \varepsilon u \right\Vert^2}_{\left\Vert \overline{u} \right\Vert^2 + 2\varepsilon \overline{u}^\top u + \varepsilon^2 \left\Vert u \right\Vert^2}
''')

st.write(r'''
  Wenn $F(\varepsilon) \rightarrow \min$ bei $\varepsilon = 0$ ist,
  so muss die Ableitung $\frac{d}{d\varepsilon} F(\varepsilon) = 0$
  für $\varepsilon = 0$ sein.
''')

st.latex(r'''
		\frac{d}{d\varepsilon} F(\varepsilon) =
		-2u^\top v + 2\overline{u}^\top u + 2\varepsilon \left\Vert u \right\Vert^2 =
		2u^\top\left(v - \overline{u} \right)  + 2\varepsilon \left\Vert u \right\Vert^2 \\
		= 2 \left\langle v - \overline{u}, u \right\rangle + 2\varepsilon \left\Vert u \right\Vert^2 \stackrel{!}{=} 0
    \quad \stackrel{\varepsilon = 0}{\Leftrightarrow} \quad
    \left\langle v - \overline{u}, u \right\rangle = 0
''')

st.write(r'''
  Der Vektor $v - \overline{u}$, also die Differenz zwischen $v$ und
  dem am nächsten gelegene Punkt $\overline{u} \in U$ steht orthogonal
  zu allen möglichen Vektoren im Raum $U$.
''')

st.write(r'''
  (ii) Weiter wird gezeigt, dass $\left\Vert v - \overline{u}\right\Vert$
  eine untere Schranke von $\left\Vert v - u\right\Vert$ ist.
''')

st.latex(r'''
  \left\Vert v - u \right\Vert^2
	= \left\Vert v - \overline{u} + \overline{u} - u \right\Vert^2
	\stackrel{\substack{
    \textrm{da }
    (v - \overline{u}) \perp (\overline{u} - u) \\
    \Rightarrow~ \textrm{Pythagoras}\\
    c^2 = a^2 + b^2
  }}{=}
  \left\Vert v - \overline{u} \right\Vert^2 + \underbrace{\left\Vert \overline{u} - u \right\Vert^2}_{\geqslant 0} \geqslant \left\Vert v - \overline{u} \right\Vert^2
''')

st.write(r'''
  $\Rightarrow\quad\overline{u}$ ist ein Minimum von $\left\Vert v - u \right\Vert^2$
''')

st.write(r'''
  (iii) Sei $u_1, ..., u_n$ eine Basis von $U$ und
  $\overline{u} = \sum_{i=1}^n \overline{\alpha}_i u_i$ so gilt
  $\forall k = 1, ..., n$ mit (i) und (ii)
''')

st.latex(r'''
	\left\langle v - \overline{u}, u_k \right\rangle
	= \left\langle v - \sum_{i=1}^n \overline{\alpha}_i u_i, u_k \right\rangle
	= \left\langle v, u_k \right\rangle -
	\sum_{i=1}^n \overline{\alpha}_i \left\langle  u_i, u_k \right\rangle = 0 \\
	\Leftrightarrow \quad
	\sum_{i=1}^n \overline{\alpha}_i \left\langle u_i, u_k \right\rangle = \left\langle v, u_k \right\rangle\\
  \Rightarrow\quad
  \underbrace{\left( \begin{array}{ccccc}
    \left\langle u_1, u_1 \right\rangle & \dots &
    \left\langle u_i, u_1 \right\rangle & \dots &
    \left\langle u_n, u_1 \right\rangle \\
    \vdots & \ddots & \vdots & \ddots & \vdots \\
    \left\langle u_1, u_k \right\rangle & \dots &
    \left\langle u_i, u_k \right\rangle & \dots &
    \left\langle u_n, u_k \right\rangle \\
    \vdots & \ddots & \vdots & \ddots & \vdots \\
    \left\langle u_1, u_n \right\rangle & \dots &
    \left\langle u_i, u_n \right\rangle & \dots &
    \left\langle u_n, u_n \right\rangle \\
  \end{array} \right)}_{
    \textrm{Gramsche Matrix }G
  } \left( \begin{array}{c}
    \overline{\alpha}_1 \\
    \vdots \\
    \overline{\alpha}_i \\
    \vdots \\
    \overline{\alpha}_n
  \end{array} \right) =
  \left( \begin{array}{c}
    \left\langle v, u_1 \right\rangle \\
    \vdots \\
    \left\langle v, u_k \right\rangle \\
    \vdots \\
    \left\langle v, u_n \right\rangle
  \end{array} \right)
''')



u1 = np.array([[2+random.random()],[0.2+random.random()]]).reshape((2,1))
u2 = np.array([[0.2+random.random()],[2+random.random()]]).reshape((2,1))
v = np.array([[1+random.random()],[1+random.random()]]).reshape((2,1))

G = np.array([[u1.T @ u1, u1.T @ u2],
              [u2.T @ u1, u2.T @ u2]]).reshape((2,2))

v_ = np.array([v.T @ u1,v.T @ u2]).reshape((2,1))
u_ = np.linalg.solve(G, v_)

u1_ = u1 * u_[0]
u2_ = u2 * u_[1]

fig, ax = plt.subplots()
ax.plot([0, v[0][0]], [0, v[1][0]], 'k', label=r'$v$')
ax.plot([0, u1[0][0]], [0, u1[1][0]], 'r--', label=r'$u_1$')
ax.plot([0, u2[0][0]], [0, u2[1][0]], 'b--', label=r'$u_2$')

ax.plot([0, u1_[0][0]], [0, u1_[1][0]], 'r', label=r'$\alpha_1 {u}_1$')
ax.plot([u1_[0][0], u1_[0][0] + u2_[0][0]], [u1_[1][0], u1_[1][0] + u2_[1][0]], 'b:')
ax.plot([0, u2_[0][0]], [0, u2_[1][0]], 'b', label=r'$\alpha_2 {u}_2$')
ax.plot([u2_[0][0], u2_[0][0] + u1_[0][0]], [u2_[1][0], u2_[1][0] + u1_[1][0]], 'r:')

ax.axis('equal')
ax.legend()
st.pyplot(fig)
st.caption(r'''
  Beispiel im zweidimensionalen Raum einer Basistransformation des Vektors
  $v_{kartesisch} = \left( \begin{array}{c} \varphi_1 \\ \varphi_2 \end{array} \right) 
  = \varphi_1\left( \begin{array}{c} 1 \\ 0 \end{array} \right) + \varphi_2\left( \begin{array}{c} 0 \\ 1 \end{array} \right)$
  in den Raum $U$ mit der Basis $u_1, u_2$ in einen Vektor $v_U = \left( \begin{array}{c} \alpha_1 \\ \alpha_2 \end{array} \right)
  = \alpha_1 u_1 + \alpha_2 u_2$.
''')
if st.button('Weiteres zufälliges Beispiel'):
    # st.rerun()
    nix = 0

st.write(r'''
  **Orthonormalsystem**
         
  Ist die Basis $u_1, ..., u_n$ ein Orthonormalsystem, so gilt
''')

st.latex(r'''
	\begin{align*}
		\left\langle u_i, u_j \right\rangle =
		\left\{ \begin{array}{ll}
			0 &\quad \textrm{für } i \neq j\\
			1 &\quad \textrm{für } i = j\\
		\end{array} \right.
	\end{align*}
''')

st.write(r'''
	und die Gramsche Matrix $G$ wird zur Einheitsmatrix
''')

st.latex(r'''
	\begin{align*}
		\left( \begin{array}{ccc}
			1 \\
			& \ddots \\
			& & 1
		\end{array} \right)
		\underbrace{\left( \begin{array}{c}
			\overline{\alpha}_1 \\
			\vdots \\
			\overline{\alpha}_n
		\end{array} \right)}_{\overline{u}} &=
		\left( \begin{array}{c}
			\left\langle v, u_1 \right\rangle \\
			\vdots \\
			\left\langle v, u_n \right\rangle
		\end{array} \right)
	\end{align*}
''')




st.stop()

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





st.subheader('Beispiel mit Datenlücke')

raw_file = os.path.join('data', 'measurement_2022-01-18T080000000_1_0xA0F5.csv')

df = pd.read_csv(raw_file, skiprows=[1])
df.describe()


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

start_i = st.slider('start_i', 0, bmX.size, 16337, 1)
# start_i = 16330
n_part = st.slider('Anzahl Punkte für die Transformation', 1, 3000, 1000, 1)
gap = st.slider('Größe der Lücke', 0, 1000, 200, 1)
gap_pos = st.slider('Position der Lücke', 0.0, 1.0, 2/3)

end_i = start_i + math.floor((n_part-gap)*gap_pos)
start_i2 = end_i + gap
end_i2 = start_i2 + math.ceil((n_part-gap)*(1-gap_pos))

t_part = t[start_i:end_i]
t_part2 = t[start_i2:end_i2]
t_part2 = t_part2 - t_part[0]
t_part = t_part - t_part[0]
t_parts = np.concatenate((t_part, t_part2))

# t_parts = t_parts - t_parts[0]

bmX_part = bmX[start_i:end_i]
bmY_part = bmY[start_i:end_i]

bmX_part2 = bmX[start_i2:end_i2]
bmY_part2 = bmY[start_i2:end_i2]

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
fig_freq_, ax_freq_ = plt.subplots()

ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), label=r'Lücke', color='tab:blue')

ax_freq_.plot(frequencies[:,0], 
        frequencies[:,1]/np.amax(frequencies[:,1]), label=r'Lücke', color='tab:blue')




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
ax_freq_.plot(frequencies[:,0], 
        frequencies[:,1]/np.amax(frequencies[:,1]), label=r'letzter Wert (fft)', color='tab:orange', alpha=0.5)



# frequencies = discrete_fourier_transformation(sig_windowed_last_value, t_part)
# ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#         frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), ':', label=r'letzter Wert (diskret)', color='tab:orange')

sig_windowed_zeros = hamming_window(signal, t_part)
sig_windowed_zeros[end_i-start_i:start_i2-start_i] = np.zeros((start_i2-end_i)).reshape((start_i2-end_i,1))
ax_raw[2].plot(t_part, sig_windowed_zeros, color='tab:green')

frequencies = fast_fourier_transformation(sig_windowed_zeros, t_part[-1] - t_part[0])
ax_freq.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1]/np.amax(frequencies[max_i-display_range:max_i+display_range,1]), label=r'mit Nullen (fft)', color='tab:green', alpha=0.5)
ax_freq_.plot(frequencies[:,0], 
        frequencies[:,1]/np.amax(frequencies[:,1]), label=r'mit Nullen (fft)', color='tab:green', alpha=0.5)


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

st.pyplot(fig_freq_)

st.stop()


# def polynomial_interpolation(points):
#   n = points.shape[0]

#   A = np.zeros((n,n))
#   b = np.zeros((n,1))
#   for i in range(0,n):
#     t = points[i][0]
#     for j in range(0, n):
#       A[i][j] = math.pow(t, j)
    
#     b[i] = points[i][1]

#   return np.linalg.solve(A, b)

# def calc_polyline(coefs, a, b, res):
#   polyline = np.zeros((res,2))
#   dim = coefs.size

#   for i in range(0,res):
#     t = a + i/(res-1) * (b - a)
#     p = 0
#     for k in range(0,dim):
#       p = p + coefs[k][0]*math.pow(t,k)

#     polyline[i][0] = t
#     polyline[i][1] = p

#   return polyline

# duration = t[end_i] - t[start_i]
# def calc_rotation_freq(f, duration):

#   fig, ax = plt.subplots()
#   ax.plot(hanning_window(f))
#   st.pyplot(fig)

#   frequencies = discrete_fourier_transformation(hanning_window(f), duration)

#   # freq = np.fft.fft(bmX_part)
#   # freq_normed = np.zeros((math.floor(freq.size/2), 4))
#   # for i in range(0,math.floor(freq.size/2)):
#   #     frequency = i/duration * 1000
#   #     norm = math.sqrt( freq.real[i]*freq.real[i] + freq.imag[i]*freq.imag[i] )
#   #     freq_normed[i][0] = frequency
#   #     freq_normed[i][1] = norm
#   #     freq_normed[i][2] = freq.real[i]
#   #     freq_normed[i][3] = freq.imag[i]

#   # fig, ax = plt.subplots()
#   # ax.plot(freq_normed[:,0], freq_normed[:,1])
#   # ax.plot(freq_normed[:,0], freq_normed[:,2])
#   # ax.plot(freq_normed[:,0], freq_normed[:,3])
#   # st.pyplot(fig)

#   max_i = np.argmax(frequencies[:,1])
#   # max_freq = frequencies[max_i][0]

#   peak_coefs = polynomial_interpolation(frequencies[max_i-1:max_i+2,[0,1]])
#   peak_polyline = calc_polyline(peak_coefs, frequencies[max_i-1,0], frequencies[max_i+1,0], 100)

#   real_coefs = polynomial_interpolation(frequencies[max_i-1:max_i+2,[0,2]])
#   real_polyline = calc_polyline(real_coefs, frequencies[max_i-1,0], frequencies[max_i+1,0], 100)

#   imag_coefs = polynomial_interpolation(frequencies[max_i-1:max_i+2,[0,3]])
#   imag_polyline = calc_polyline(imag_coefs, frequencies[max_i-1,0], frequencies[max_i+1,0], 100)

#   # set first derivative to zero and reolve to x 
#   # p(x) = a_0 + x*a_1 + x^2*a_2
#   # p'(x) = a_1 + 2*x*a_2 = 0  =>  x = - a_1 / (2*a_2)
#   peak_freq = - peak_coefs[1] / (2*peak_coefs[2])
#   peak_value = peak_coefs[0] + peak_coefs[1]*peak_freq + peak_coefs[2]*peak_freq*peak_freq

#   real_value = real_coefs[0] + real_coefs[1]*peak_freq + real_coefs[2]*peak_freq*peak_freq
#   imag_value = imag_coefs[0] + imag_coefs[1]*peak_freq + imag_coefs[2]*peak_freq*peak_freq

#   display_range = 5
#   fig, ax = plt.subplots()
#   ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#           frequencies[max_i-display_range:max_i+display_range,1], label=r'Norm')
#   ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#           frequencies[max_i-display_range:max_i+display_range,2], label=r'Real')
#   ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
#           frequencies[max_i-display_range:max_i+display_range,3], label=r'Imag')
#   ax.plot(peak_polyline[:,0], peak_polyline[:,1])
#   ax.plot(peak_freq, peak_value, '.')
#   ax.plot(real_polyline[:,0], real_polyline[:,1])
#   ax.plot(peak_freq, real_value, '.')
#   ax.plot(imag_polyline[:,0], imag_polyline[:,1])
#   ax.plot(peak_freq, imag_value, '.')
#   ax.legend()
#   st.pyplot(fig)

#   return np.array([peak_freq, peak_value, real_value, imag_value])



# peakX_freq = calc_rotation_freq(bmX_part, duration)
# peakY_freq = calc_rotation_freq(bmY_part, duration)

# # st.write(peakX_freq, peakY_freq)

# peak_freq = (peakX_freq[0] + peakY_freq[0]) / 2
# delta_t = 1000 / peak_freq

# t_cutted = np.zeros((n_part,1))
# for i in range(0,n_part):
#   t_cutted[i] = (t_part[i] % delta_t) / delta_t * 2 * math.pi




# def trig_approx(f, t, K = 3):
#   n = f.size
  
#   A = np.zeros([n,2*K+1])
#   for i in range(0,n):
#     A[i,0] = 1 / 2
#     for k in range(0,K):
#       A[i,1+k] = math.cos((1 + k) * t[i])
#       A[i,1+K+k] = math.sin((1 + k) * t[i])

#   b = np.reshape(f, [n,1])
#   return np.linalg.solve(A.T @ A, A.T @ b)

# def eval_trig_approx(coefs, t_i):
#   K = math.floor(coefs.size/2)

#   a = np.zeros([K,1])
#   b = np.zeros([K,1])

#   for k in range(0,K):
#     a[k] = coefs[1+k]
#     b[k] = coefs[1+K+k]

#   f_i = coefs[0]/2

#   for k in range(0,K):
#     f_i = f_i + a[k] * math.cos((1 + k) * t_i)
#     f_i = f_i + b[k] * math.sin((1 + k) * t_i)

#   return f_i


# res = 100

# def calc_trig_line(coefs, a, b, res = 100):
#   f = np.zeros([res,1])
#   i = 0
#   for t in np.linspace(a, b, res):
#     f[i] = eval_trig_approx(coefs, t)
#     i = i + 1

#   return f

# coefsX = trig_approx(bmX_part, t_cutted, 1)
# approxX = calc_trig_line(coefsX, 0, 2*math.pi, res)

# coefsY = trig_approx(bmY_part, t_cutted, 1)
# approxY = calc_trig_line(coefsY, 0, 2*math.pi, res)

# roundabout = np.linspace(0, 2*math.pi, res)

# fig, ax = plt.subplots()
# ax.plot(t_cutted, bmX_part, '.')
# ax.plot(t_cutted, bmY_part, '.')
# ax.plot(roundabout, approxX)
# ax.plot(roundabout, approxY)
# st.pyplot(fig)



# phiX = math.atan2(coefsX[1], coefsX[2])
# if phiX < 0:
#   phiX = phiX * -1
# else:
#   phiX = 2*math.pi - phiX

# # phiX = phiX - math.pi/2

# # approxX_max_i = np.argmax(approxX)
# # st.write(roundabout[approxX_max_i]-math.pi/2, phiX)

# phiY = math.atan2(coefsY[1], coefsY[2])
# if phiY < 0:
#   phiY = phiY * -1
# else:
#   phiY = 2*math.pi - phiY

# approxX_shifted = calc_trig_line(coefsX, phiX, 2*math.pi + phiX, res)
# approxY_shifted = calc_trig_line(coefsY, phiY, 2*math.pi + phiY, res)

# tX_cutted = np.zeros((n_part,1))
# tY_cutted = np.zeros((n_part,1))
# for i in range(0,n_part):
#   tX_cutted[i] = ((t_part[i] % delta_t) / delta_t * 2 * math.pi - phiX) % (2*math.pi)
#   tY_cutted[i] = ((t_part[i] % delta_t) / delta_t * 2 * math.pi - phiY) % (2*math.pi)

# fig, ax = plt.subplots()
# ax.plot(tX_cutted, bmX_part, '.')
# ax.plot(tY_cutted, bmY_part, '.')
# ax.plot(roundabout, approxX_shifted)
# ax.plot(roundabout, approxY_shifted)
# st.pyplot(fig)

# tX_sorted_ind = np.argsort(tX_cutted.flatten(), axis=0)
# tY_sorted_ind = np.argsort(tY_cutted.flatten(), axis=0)

# bmX_rearranged = np.take_along_axis(bmX_part.flatten(), tX_sorted_ind, axis=0) 
# bmY_rearranged = np.take_along_axis(bmY_part.flatten(), tY_sorted_ind, axis=0) 

# # tX_i_min = np.argmin(tX_cutted)
# # tY_i_min = np.argmin(tY_cutted)

# # bmX_rearranged = np.zeros((n_part,1))
# # bmY_rearranged = np.zeros((n_part,1))
# # for i in range(0,n_part):
# #   bmX_rearranged[i] = bmX_part[(i+tX_i_min) % n_part]
# #   bmY_rearranged[i] = bmY_part[(i+tY_i_min) % n_part]

# fig, ax = plt.subplots()
# ax.plot(bmX_part, bmY_part, '.')
# ax.plot(bmX_rearranged, bmY_rearranged, 'x')
# st.pyplot(fig)

# # tX_cutted = np.zeros((n_part,1))
# # tY_cutted = np.zeros((n_part,1))

# # waveX_part = np.zeros((n_part,1))
# # waveY_part = np.zeros((n_part,1))

# # sin_part = np.zeros((n_part,1))
# # cos_part = np.zeros((n_part,1))

# # # https://en.wikibooks.org/wiki/Trigonometry/Simplifying_a_sin(x)_%2B_b_cos(x)
# # ampX = math.sqrt(peakX_freq[2]*peakX_freq[2] + peakX_freq[3]*peakX_freq[3])
# # phiX = math.atan2(peakX_freq[2], peakX_freq[3])

# # ampY = math.sqrt(peakY_freq[2]*peakY_freq[2] + peakY_freq[3]*peakY_freq[3])
# # phiY = math.atan2(peakY_freq[2], peakY_freq[3]) + math.pi/2

# # st.write(phiX, phiY)

# # max_bm = np.amax(bmX_part)
# # for i in range(0,n_part):
# #     t_i = t_part[i] % delta_t
# #     phi_i = t_i / delta_t * 2 * math.pi
# #     t_cutted[i] = t_i

# #     waveX_part[i] = peakX_freq[2]*math.cos(phi_i) + peakX_freq[3]*math.sin(phi_i)
# #     waveY_part[i] = peakY_freq[2]*math.cos(phi_i) + peakY_freq[3]*math.sin(phi_i)

# #     # tX_i = (t_part[i] + (phiX / (2*math.pi) * delta_t) ) % delta_t
# #     # tX_cutted[i] = tX_i
# #     # phiX_i = tX_i / delta_t * 2 * math.pi
    
# #     # sin_part[i] = ampX * math.sin(phiX_i)

# #     # tY_i = (t_part[i] + (phiY / (2*math.pi) * delta_t)) % delta_t
# #     # # tY_i = (t_part[i]) % delta_t
# #     # tY_cutted[i] = tY_i
# #     # phiY_i = tY_i / delta_t * 2 * math.pi
    
# #     # cos_part[i] = ampY * math.sin(phiY_i + math.pi/2)

# # st.write(23.34324312 % 10.43112)

# # st.write(t_cutted)

# # # tX_i_min = np.argmin(tX_cutted)
# # # tY_i_min = np.argmin(tY_cutted)

# # # bmX_rearranged = np.zeros((n_part,1))
# # # bmY_rearranged = np.zeros((n_part,1))
# # # for i in range(0,n_part):
# # #   bmX_rearranged[i] = bmX_part[(i+tX_i_min) % n_part]
# # #   bmY_rearranged[i] = bmY_part[(i+tY_i_min) % n_part]

# # # fig, ax = plt.subplots()
# # # ax.plot(tX_cutted, bmX_part, '.')
# # # ax.plot(tY_cutted, bmY_part, '.')
# # # ax.plot(tX_cutted, sin_part, '.')
# # # ax.plot(tY_cutted, cos_part, '.')
# # # st.pyplot(fig)

# # fig, ax = plt.subplots()
# # ax.plot(t_cutted, bmX_part, '.')
# # ax.plot(t_cutted, bmY_part, '.')
# # ax.plot(t_cutted, waveX_part, '.')
# # ax.plot(t_cutted, waveY_part, '.')
# # st.pyplot(fig)

# # # fig, ax = plt.subplots()
# # # ax.plot(bmX_part, bmY_part, '.')
# # # ax.plot(bmX_rearranged, bmY_rearranged, 'x')
# # # st.pyplot(fig)


# exit()

# st.header('Polynomaproximation')

# path_to_data = os.path.join('data', 'aggregated_data_export_20230315T112419880')

# csv_list = os.listdir(path_to_data)

# data = np.genfromtxt(os.path.join(path_to_data, csv_list[0]), delimiter=',')

# t = data[:,0]
# bm = data[:,21]

# K = st.slider('Grad des Polynoms', 1, 100, 30)
# n = bm.size
# bm = np.reshape(bm, [n,1])

# t_max = np.amax(t)

# # fill in Vandermonde matrix
# V_poly = np.zeros([n,K])
# for i in range(0,n):
#   for j in range(0,K):
#     V_poly[i][j] = np.power(t[i]/t_max,j)

# # method of least square
# A = V_poly.T @ V_poly
# b = V_poly.T @ bm

# # solve
# coefs = np.linalg.solve(A, b)
# # coefs = QR_solver(A, b)

# def eval_poly(t, coefs):
#   res = 0
#   for i in range(0,len(coefs)):
#     res = res + coefs[i] * np.power(t,i)
#   return res

# alpha = 0.98
# delta_t = 0

# bm_interp = np.zeros([n,1])
# for i in range(0,n):
#   t_i = i/n
#   bm_interp[i] = eval_poly(max(0, min(1, alpha * (t_i + delta_t))), coefs)

# st.line_chart(pd.DataFrame(np.array([bm[:,0], bm_interp[:,0]]).T))

# # st.line_chart(pd.DataFrame([bm, bm_interp]))