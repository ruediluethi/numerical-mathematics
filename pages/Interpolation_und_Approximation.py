import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt



st.title('Approximation')

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

st.write(r'''
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
  Sei $f$ mit $f_1, ..., f_n$ ein Vektor mit dem diskrete Messsignal
  der Funktion $f(t)$ für welche die bestmögliche Approximation
  mittels einer endlichen trigonometrischen Reihe gesucht wird.
  Es wird also ein Koeffizient $c_k$ zum
  dazugehörigen Basisvektor $s_k = e^{i \frac{2\pi k}{p}x}$ gesucht.
''')

st.latex(r'''
  c_k
  = \left\langle f, s_k \right\rangle
  = \int_0^p f(x) \cdot \overline{e^{i \frac{2\pi k}{p}x}} ~dx
  \stackrel{\substack{\textrm{Diskret-}\\\textrm{isieren}}}{=}
  \frac{1}{n} \sum_{l=0}^n f_l \cdot \overline{e^{i \frac{2\pi k }{p}l}}
''')

st.write(r'''
  aus $c_k \in \mathbb{C}$ lassen sich $a_k, b_k \in \R$ bestimmen
''')

st.latex(r'''
  \begin{align*}
    a_k = 2~\textrm{Re}(c_k) &= \frac{2}{p} \int_0^p f(x) \cos\left( \frac{2\pi k x}{p} \right) ~dx \\
    b_k = -2~\textrm{Im}(c_k) &= -\frac{2}{p} \int_0^p f(x) \sin\left( \frac{2\pi k x}{p} \right) ~dx
  \end{align*}
''')


def discrete_fourier_transformation(f, duration):
  n = f.size

  A = np.zeros((n,n))
  B = np.zeros((n,n))

  for k in range(0,n):
    for i in range(0,n):
      A[i][k] = 2/n*math.cos(2*math.pi*k*i/n)
      B[i][k] = -2/n*math.sin(2*math.pi*k*i/n)

  f = f.reshape((n,1))
  a = A @ f # real part
  b = B @ f # imaginary part

  frequencies = np.zeros((math.floor(n/2),4))
  # for (let i = 3/*Math.floor(n*0.01)*/; i < n/2; i++){
  for i in range(0,math.floor(n/2)):
    frequency = i/duration * 1000
    norm = math.sqrt( a[i][0]*a[i][0] + b[i][0]*b[i][0] )
    frequencies[i][0] = frequency
    frequencies[i][1] = norm
    frequencies[i][2] = a[i][0]
    frequencies[i][3] = b[i][0]

  return frequencies



raw_file = os.path.join('data', 'measurement_2022-01-18T080000000_1_0xA0F5.csv')

df = pd.read_csv(raw_file, skiprows=[1])
df.describe()


bmX = df['Bm X'].to_numpy()

t = np.array([])
for index, row in df.iterrows():
    t = np.append(t, pd.Timestamp(row['time']).timestamp() * 1000)

fig, ax = plt.subplots()
ax.plot(bmX)
st.pyplot(fig)

n_part = 1000
start_i = 30000
end_i = start_i + n_part

duration = t[end_i] - t[start_i]

t_part = t[start_i:end_i]
bmX_part = bmX[start_i:end_i]

frequencies = discrete_fourier_transformation(bmX_part, duration)

# freq = np.fft.fft(bmX_part)
# freq_normed = np.zeros((math.floor(freq.size/2), 4))
# for i in range(0,math.floor(freq.size/2)):
#     frequency = i/duration * 1000
#     norm = math.sqrt( freq.real[i]*freq.real[i] + freq.imag[i]*freq.imag[i] )
#     freq_normed[i][0] = frequency
#     freq_normed[i][1] = norm
#     freq_normed[i][2] = freq.real[i]
#     freq_normed[i][3] = freq.imag[i]

# fig, ax = plt.subplots()
# ax.plot(freq_normed[:,0], freq_normed[:,1])
# ax.plot(freq_normed[:,0], freq_normed[:,2])
# ax.plot(freq_normed[:,0], freq_normed[:,3])
# st.pyplot(fig)

max_i = np.argmax(frequencies[:,1])
max_freq = frequencies[max_i][0]


def polynomial_interpolation(points):
  n = points.shape[0]
  st.write(points)
  st.write(points.shape)
  st.write(n)

  A = np.zeros((n,n))
  b = np.zeros((n,1))
  for i in range(0,n):
    t = points[i][0]
    for j in range(0, n):
      A[i][j] = math.pow(t, j)
    
    b[i] = points[i][1]

  return np.linalg.solve(A, b)

def calc_polyline(coefs, a, b, res):
  polyline = np.zeros((res,2))
  dim = coefs.size

  for i in range(0,res):
    t = a + i/(res-1) * (b - a)
    p = 0
    for k in range(0,dim):
      p = p + coefs[k][0]*math.pow(t,k)

    polyline[i][0] = t
    polyline[i][1] = p

  return polyline

peak_coefs = polynomial_interpolation(frequencies[max_i-1:max_i+2,0:2])
polyline = calc_polyline(peak_coefs, frequencies[max_i-1,0], frequencies[max_i+1,0], 100)

# set first derivative to zero and reolve to x 
# p(x) = a_0 + x*a_1 + x^2*a_2
# p'(x) = a_1 + 2*x*a_2 = 0  =>  x = - a_1 / (2*a_2)
peak_freq = - peak_coefs[1] / (2*peak_coefs[2])
peak_value = peak_coefs[0] + peak_coefs[1]*peak_freq + peak_coefs[2]*peak_freq*peak_freq

st.write(max_freq, peak_freq[0])

display_range = 3
fig, ax = plt.subplots()
ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,1], label=r'Norm')
ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,2], label=r'Real')
ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
        frequencies[max_i-display_range:max_i+display_range,3], label=r'Imag')
ax.plot(polyline[:,0], polyline[:,1])
ax.plot(peak_freq,peak_value, '.')
ax.legend()
st.pyplot(fig)

delta_t = 1000 / peak_freq

bmX_cutted = np.zeros((n_part,2))
for i in range(0,n_part):
    bmX_cutted[i][0] = t_part[i] % delta_t
    bmX_cutted[i][1] = bmX_part[i]

fig, ax = plt.subplots()
ax.plot(bmX_cutted[:,0], bmX_cutted[:,1], '.')
st.pyplot(fig)





st.header('Polynomaproximation')

path_to_data = os.path.join('data', 'aggregated_data_export_20230315T112419880')

csv_list = os.listdir(path_to_data)

data = np.genfromtxt(os.path.join(path_to_data, csv_list[0]), delimiter=',')

t = data[:,0]
bm = data[:,21]

K = st.slider('Grad des Polynoms', 1, 100, 30)
n = bm.size
bm = np.reshape(bm, [n,1])

t_max = np.amax(t)

# fill in Vandermonde matrix
V_poly = np.zeros([n,K])
for i in range(0,n):
  for j in range(0,K):
    V_poly[i][j] = np.power(t[i]/t_max,j)

# method of least square
A = V_poly.T @ V_poly
b = V_poly.T @ bm

# solve
coefs = np.linalg.solve(A, b)
# coefs = QR_solver(A, b)

def eval_poly(t, coefs):
  res = 0
  for i in range(0,len(coefs)):
    res = res + coefs[i] * np.power(t,i)
  return res

alpha = 0.98
delta_t = 0

bm_interp = np.zeros([n,1])
for i in range(0,n):
  t_i = i/n
  bm_interp[i] = eval_poly(max(0, min(1, alpha * (t_i + delta_t))), coefs)

st.line_chart(pd.DataFrame(np.array([bm[:,0], bm_interp[:,0]]).T))

# st.line_chart(pd.DataFrame([bm, bm_interp]))