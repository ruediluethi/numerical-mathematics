import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd



st.title('Monte Carlo Simulation')

n = st.slider('Anzahl Punkte', 1, 10000, 1000)

fig, ax = plt.subplots()
q = 0
progress = st.progress(0)
for i in range(0, n):
    progress.progress(i/n)
    x = np.random.rand()
    y = np.random.rand()
    if x**2 + y**2 <= 1:
        ax.plot(x, y, 'b.')
        q += 1
    else:
        ax.plot(x, y, 'r.')
ax.set_aspect('equal')
st.pyplot(fig)
progress.empty()
st.write(q/n*4)

st.write(r'''
    Sei $A = [0, a]$ mit $a \leq 1$ ein Intervall, 
    $X \in [0,1]$ eine gleichverteilte Zufallsvariable mit der [Dichtefunktion](https://de.wikipedia.org/wiki/Dichtefunktion) $p(x) = 1$
    und $\mathbb{I}_A: \mathbb{R} \rightarrow \{0,1\}$ die [Indikatorfunktion](https://de.wikipedia.org/wiki/Indikatorfunktion) von $A$ mit
''')
st.latex(r'''
    \mathbb{I}_A(x) = \begin{cases}
        1 & \text{wenn } x \in A \\
        0 & \text{wenn } x \notin A
    \end{cases}
''')

st.write(r'Damit gilt für den Erwartungswert $\mathbb{E}(\mathbb{I}_A(X))$')

# = \sum_{i = 1}^n x_i \cdot \mathbb{P}(X = x_i)
st.latex(r'''
    \mathbb{E}(\mathbb{I}_A(X))
    = \int_{-\infty}^\infty \mathbb{I}_A(x) \cdot \underbrace{p(x)}_{=1} ~dx
    = \int_{0}^a 1 ~dx = a
''')

st.write(r'''
    Sei $M = \left[0,\infty\right) \times \left[-\pi,\pi\right]$
    und sei $T: M \rightarrow \mathbb{R}^2$ mit $T(r, \varphi) = (r \cos \varphi, r \sin \varphi)^\top$ die Polarkoordinatentransformation
    mit der [Funktionaldeterminante](https://de.wikipedia.org/wiki/Funktionaldeterminante) $\det T'$
''')
st.latex(r'''
    T'(r,\varphi) = \left( \begin{array}{cc}
		\cos \varphi & -r \sin \varphi \\
		\sin \varphi & r \cos \varphi		
	\end{array} \right) \quad\Rightarrow\quad
	\det T'(r,\varphi) = r \cos^2 \varphi + r \sin^2 \varphi = r \neq 0      
''')

st.write(r'So gilt für die Fläche $T(M)$')
st.latex(r'''
    \int_{T(M)} 1 ~d(x, y) 
    = \int_M 1 \cdot \det T' (r,\varphi) ~d(r, \varphi) 
    = \int_{\left[-\pi,\pi\right]} \int_{\left[ 0, 1 \right]} r ~dr ~d\varphi
''')

st.write(r'''
    Seien $X, Y \in [0,1]$ zwei unabhängige gleichverteilte Zufallsvariablen mit der gemeinsamen Dichtefunktion $p(x, y) = 1$
    und $\mathbb{I}_{Q}: \mathbb{R}^2 \rightarrow \{0,1\}$ die Indikatorfunktion für die Viertelkreisfläche mit Radius $1$
''')
st.latex(r'''
    \mathbb{I}_Q(x, y) = \begin{cases}
        1 & \text{wenn } x^2 + y^2 \leq 1 \text{ und } x, y \geq 0\\
        0 & \text{sonst}
    \end{cases}
''')

st.write(r'Damit gilt für den Erwartungswert $\mathbb{E}(\mathbb{I}_Q(X, Y))$')
st.latex(r'''
    \mathbb{E}(\mathbb{I}_Q(X, Y))
    = \int_{-\infty}^\infty \int_{-\infty}^\infty \mathbb{I}_Q(x, y) \cdot p(x, y) ~d(x, y) 
    = \int_{T(M)} \mathbb{I}_Q(x, y) \cdot 1 ~d(x, y) \\
    = \int_M \underbrace{\mathbb{I}_Q\left(T(r, \varphi)\right)}_{=1 \text{ für } 0 \leq \varphi \leq \frac{\pi}{2}} \cdot \underbrace{\det T' (r,\varphi)}_{=r} ~d(r, \varphi) 
    = \int_{\left[0,\frac{\pi}{2}\right]} \int_{\left[ 0, 1 \right]} r ~dr ~d\varphi
    = \int_{\left[0,\frac{\pi}{2}\right]} \left[ \frac{r^2}{2} \right]_{0}^{1} ~d\varphi
    = \frac{\pi}{4}
''')
