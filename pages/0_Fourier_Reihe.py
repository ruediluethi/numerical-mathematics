import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

st.title('Die Fourier-Reihe')

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

st.write('Es gilt')
st.latex(r'''
    a \cdot \cos\left( \varphi \right) + b \cdot \sin\left( \varphi \right)
    = c \cdot \sin\left( \varphi + \Delta\varphi \right)
''')
st.write('mit')
st.latex(r'''
    c = \sqrt{a^2 + b^2} \qquad\text{und}\qquad \Delta\varphi = \arctan\frac{b}{a}
''')
st.write('Weil')
st.latex(r'''
    c \cdot \sin\left( \varphi + \Delta\varphi \right) \stackrel{\substack{\text{Additions-}\\\text{theoreme}}}{=}
    \underbrace{c \cdot \cos\Delta\varphi}_{= a} \cos{\left( \varphi \right)}  + \underbrace{c \cdot \sin\Delta\varphi}_{= b} \sin{\left( \varphi \right)}
''')
st.write('Daraus folgt')
st.latex(r'''
    (1)\qquad a = c \cdot \cos\Delta\varphi \quad\Leftrightarrow\quad c = \frac{a}{\cos\Delta\varphi} \\
    (2)\qquad b = c \cdot \sin\Delta\varphi \quad\Leftrightarrow\quad c = \frac{b}{\sin\Delta\varphi} \\
    \Rightarrow \qquad \frac{a}{\cos\Delta\varphi} = \frac{b}{\sin\Delta\varphi} \stackrel{\tan=\frac{\sin}{\cos}}{=} \frac{a}{b} = \tan\Delta\varphi
    \quad \Leftrightarrow \quad \Delta\varphi = \arctan \frac{b}{a}
''')
st.write('Weiter gilt')
st.latex(r''' 
    \sqrt{a^2 + b^2} \stackrel{(1), (2)}{=} \sqrt{\left(c \cdot \cos\Delta\varphi\right)^2 + \left(c \cdot \sin\Delta\varphi\right)^2} 
    = c \cdot \underbrace{\sqrt{\cos^2\Delta\varphi + \sin^2\Delta\varphi}}_{=1} = c
''')

n = 500
phi = np.linspace(0, 2 * np.pi, n)
phi_long = np.linspace(-np.pi, 3 * np.pi, n)

a = st.slider('a', -2.0, 2.0, 1.0)
b = st.slider('b', -2.0, 2.0, 1.0)

c = math.sqrt(a**2 + b**2)
delta_phi = math.atan2(a, b)

fig, ax = plt.subplots()
ax.plot(phi_long, a*np.cos(phi_long), ':', color='tab:blue')
ax.plot(phi, a*np.cos(phi), color='tab:blue', label=r'$a\cdot\cos(\varphi)$')
ax.plot(phi_long, b*np.sin(phi_long), ':', color='tab:orange')
ax.plot(phi, b*np.sin(phi), color='tab:orange', label=r'$b\cdot\sin(\varphi)$')
ax.plot(phi_long, a*np.cos(phi_long)+b*np.sin(phi_long), ':', color='tab:red')
ax.plot(phi, a*np.cos(phi)+b*np.sin(phi), color='tab:red', label=r'$a\cdot\cos(\varphi) + b\cdot\sin(\varphi)$')
ax.plot(phi_long, c*np.sin(phi_long), ':', color='tab:purple', label=r'$c\cdot\sin(\varphi)$')
ax.plot(phi, c*np.sin(phi+delta_phi), '--', color='tab:purple', label=r'$c\cdot\sin(\varphi + \Delta\varphi)$')

ax.plot([np.pi/2-delta_phi, np.pi/2-delta_phi], [0, c], 'k-.', label=r'$c$')
ax.plot([np.pi/2-delta_phi, np.pi/2], [c, c], 'k--', label=r'$\Delta\varphi$')


ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

st.pyplot(fig)

grade = st.slider('Grad der Fourier Reihe', 1, 30, 5)
n_cut = st.slider('Anzahl der Schneiden', 1, 10, 4)

C_x = np.zeros(grade)
C_y = np.zeros(grade)
phase_x = np.zeros(grade)
phase_y = np.zeros(grade)

fig, ax = plt.subplots()

f_x = np.zeros(n)
f_y = np.zeros(n)
with st.sidebar:
    edit_manually = st.checkbox("manuell editieren", value=False)
    for k in range(grade):
        default_c = 0.0
        # if k == 0:
        #     default_c = 1.0
        # if k == 4:
        #     default_c = 0.2
        if k%n_cut == 0:
            default_c = 1.0 / ((k/n_cut+1)**2)

        # st.write(default_c)

        if edit_manually:
            st.subheader(f'k={k+1}')
            c_k = st.slider(f'c_{k}', 0.0, 2.0, default_c, key=f'c_{k}')
        else:
            c_k = default_c
        # C_x[k] = st.slider(f'C_x_{k}', -2.0, 2.0, default_c, key=f'C_x_{k}')
        # C_y[k] = st.slider(f'C_y_{k}', -2.0, 2.0, default_c, key=f'C_y_{k}')
        C_x[k] = c_k
        C_y[k] = c_k

        # phase_x[k] = st.slider(f'phase_x_{k}', -2.0, 2.0, 0.0, key=f'phase_x_{k}')
        # phase_y[k] = st.slider(f'phase_y_{k}', -2.0, 2.0, np.pi/2, key=f'phase_y_{k}')
        if edit_manually:
            phase_k = st.slider(f'phase_{k}', -2.0, 2.0, 0.0, key=f'phase_{k}')
        else:
            phase_k = 0.0
        phase_x[k] = phase_k
        phase_y[k] = phase_k + np.pi/2

        f_x_k = C_x[k]*np.sin(phi*(k+1) + phase_x[k])
        f_y_k = C_y[k]*np.sin(phi*(k+1) + phase_y[k])

        ax.plot(phi, f_x_k, color='tab:blue', alpha=0.3)
        ax.plot(phi, f_y_k, color='tab:orange', alpha=0.3)

        f_x += f_x_k
        f_y += f_y_k

ax.plot(phi, f_x, color='tab:blue')
ax.plot(phi, f_y, color='tab:orange')

st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_title("spike_polar plot")
ax.set_aspect('equal')
ax.plot(f_x, f_y, 'k')
st.pyplot(fig)


# berechnen des betrages
f_xy = np.sqrt(f_x**2 + f_y**2)

# Funktion welche exakt so auch in der Fusion Software implementiert ist
def trig_approx(f: np.ndarray, t: np.ndarray, K: int = 20) -> np.ndarray:
    """
    Approximates a given function using trigonometric series.

    Args:
        f (np.ndarray): The function values.
        t (np.ndarray): The 'time' values normalized as radians (0 to 2pi).
        K (int, optional): The order of the approximation. Defaults to 20.

    Returns:
        np.ndarray: The coefficients of the trigonometric series as complex array. first element is a0/2, then a1, i*b1, a2, i*b2, ...
    """
    n = len(f)
    k = np.arange(1, K + 1)
    A = np.zeros([n, 2*K + 1])
    A[:, 0] = 1
    A[:, 1:K+1] = np.cos(k * t[:, None])
    A[:, K+1:] = np.sin(k * t[:, None])
    b = np.reshape(f, [n, 1])
    coeffs = np.linalg.solve(A.T @ A, A.T @ b).flatten()
    # store the coefficients in a complex array. first element is a0/2, then a1, i*b1, a2, i*b2, ...
    a = coeffs[0:K+1]
    b = coeffs[K+1:2*K+1]
    b = np.insert(b, 0, 0)
    return a + 1j*b


coeffs = trig_approx(f_xy, phi, K=grade*2)

fig_bar, ax_bar = plt.subplots()
ax_bar.bar(np.arange(len(coeffs)), np.abs(coeffs))
ax_bar.set_xlabel('Koeffizienten-Index')
ax_bar.set_ylabel('Betrag')
ax_bar.set_title('Beträge der komplexen Fourier-Koeffizienten')
st.pyplot(fig_bar)


fig, ax = plt.subplots()
ax.plot(phi, f_xy, 'k-', label='Originalfunktion')

f_xy_approx = np.zeros(n)
for k in range(grade*2):
    f_xy_k = np.real(coeffs[k])*np.cos(phi*k) + np.imag(coeffs[k])*np.sin(phi*k)
    f_xy_approx += f_xy_k
    ax.plot(phi, f_xy_k, color='tab:blue', alpha=0.3, label=f'{k}-ter Term')
    

ax.plot(phi, f_xy_approx, '--', color='tab:orange', label='Summe')

ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
st.pyplot(fig)