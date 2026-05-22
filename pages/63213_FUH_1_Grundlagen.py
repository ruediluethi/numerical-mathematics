import streamlit as st
import numpy as np
import math 
from matplotlib import pyplot as plt
import plotly.graph_objects as go

st.title("1. Grundlagen")

st.header("1.2.1 Topologie")

st.subheader("1.2.1.1 Übungsaufgabe zylindrische Schraubenfeder")


st.write(r'''
    Sei $p(t): \mathbb{R} \rightarrow \mathbb{R}^3$ ein durch $t \in [0...1]$ parametrisierter Pfad mit
    $x(t), y(t), z(t)$ als Komponenten:     
''')

st.latex(r'''
    p(t) = \left( \begin{array}{c}
        x(t) \\
        y(t) \\
        z(t)
    \end{array} \right) 
''')

st.write(r'So ist die Länge des Pfades durch folgendes Integral definiert:')
st.latex(r'''
    L = \int_0^1 \sqrt{x'(t)^2 + y'(t)^2 + z'(t)^2} ~dt
''')

st.write(r'Sei nun $x(t) = r \cos(t), y(t) = r \sin(t), z(t) = \frac{t}{2\pi}h$ mit Höhe $h$ und $t \in [0...2\pi]$.')
st.write(r"So gilt für $p'(t)$:")

st.latex(r'''
    p'(t) = \left( \begin{array}{c}
        -r \sin(t) \\
        r \cos(t) \\
        \frac{h}{2\pi}
    \end{array} \right) 
''')

st.write(r'und damit gilt für die Länge:')

st.latex(r'''
    L = \int_0^{2\pi} \sqrt{\underbrace{\left(-r\sin(t)\right)^2 + \left(r\cos(t)\right)^2}_{=r^2} + \left( \frac{h}{2\pi} \right)^2} ~dt
    = \sqrt{ r^2 + \left( \frac{h}{2\pi} \right)^2} \int_0^{2\pi} 1 dt \\
    = 2\pi\sqrt{ r^2 + \left( \frac{h}{2\pi} \right)^2} = \sqrt{ 4\pi^2r^2 + h^2} = L
''')

st.write(r'Abhängigkeit des Radius $r$ zur Höhe $h$')
st.latex(r'''
    \Leftrightarrow \quad 4\pi^2r^2 + h^2 = L^2
    \quad \Leftrightarrow \quad r = \frac{\sqrt{L^2 - h^2}}{2\pi} = r(h)
''')



rotations = 2
t = np.linspace(0, 2*np.pi * rotations, 100)

L = 1
fig = go.Figure()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
colors = ['#26547D', '#EF436B', '#FFCE5C', '#05C793']
n = 7
for i in range(0,n+1):
    h = 1 - (i/n)**2
    r = np.sqrt(L**2 - h**2) / (2*np.pi)
    t_ = t# + i * 2*np.pi
    x = r * np.cos(t_)
    y = r * np.sin(t_)
    z = h * t_ / (2*np.pi)

    # ax.plot(x, y, z + 0.5 - h/2, alpha=1-h*0.9)
    rgb = tuple(int(colors[i % len(colors)].lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z - h*rotations/2, mode='lines', 
                               line=dict(color=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},1.0)', width=(1-i/n)*5+1),
                               showlegend=False))


st.plotly_chart(fig)

st.header("1.2.2 Graphentheorie")

st.subheader("1.2.2.2 Übungsaufgabe")

n = st.slider('Anzahl Punkte', 1, 1000, 100)

S = np.zeros((n,2))
for i in range(0, n):
    S[i,0] = np.random.rand()
    S[i,1] = np.random.rand()

fig, ax = plt.subplots()
ax.plot(S[:,0], S[:,1], 'k.')

for i in range(0, n):
    min_j = -1
    min_dist = np.sqrt(2)
    for j in range(0, n):
        if i == j:
            continue
        dx = S[i,0] - S[j,0]
        dy = S[i,1] - S[j,1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist < min_dist:
            min_dist = dist
            min_j = j

    # st.write(i, min_j)

    ax.plot([S[i,0], S[min_j,0]], [S[i,1], S[min_j,1]], 'k-')

ax.set_aspect('equal')
st.pyplot(fig)