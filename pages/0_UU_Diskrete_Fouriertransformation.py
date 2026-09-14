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

st.page_link('pages/Experiment_DFT_mit_Luecke.py', label='Experiment mit einer dummy Sinus-Funktion und einer Lücke', icon='🧪')

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
if st.button('Weiteres zufälliges Beispiel'):
  nix = 0

st.pyplot(fig)
st.caption(r'''
  Beispiel im zweidimensionalen Raum einer Basistransformation des Vektors
  $v_{kartesisch} = \left( \begin{array}{c} \varphi_1 \\ \varphi_2 \end{array} \right) 
  = \varphi_1\left( \begin{array}{c} 1 \\ 0 \end{array} \right) + \varphi_2\left( \begin{array}{c} 0 \\ 1 \end{array} \right)$
  in den Raum $U$ mit der Basis $u_1, u_2$ in einen Vektor $v_U = \left( \begin{array}{c} \alpha_1 \\ \alpha_2 \end{array} \right)
  = \alpha_1 u_1 + \alpha_2 u_2$.
''')
  

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



