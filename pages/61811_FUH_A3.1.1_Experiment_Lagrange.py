import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import pandas as pd

st.title('Optimierungsproblem unter Nebenbedingungen')

st.latex(r'''
    \min_{x,y \in \mathbb{R}} f(x,y) = (x - 2)^2 + 2(y-1)^2 \\
    g_1(x,y) = x + 4y \leq 3 \\
    g_2(x,y) = -x + y \leq 0
''')

st.write('Globales Minimum')
st.latex(r'''
    \frac{\partial f}{\partial x} = 2(x - 2) \stackrel{!}{=} 0 
    \Leftrightarrow x = 2 \\
    \frac{\partial f}{\partial y} = 4(y - 1) \stackrel{!}{=} 0
    \Leftrightarrow y = 1 \\
''')
f_minx = 2
f_miny = 1
f_minf = (f_minx - 2)**2 + 2*(f_miny - 1)**2

st.write(r'Minimum auf dem Rand von $g_1$')
st.latex(r'''
    \partial g_1: h_1(t) = \frac{1}{4}(3 - t) \\
    f(t, h_1(t)) = f_1(t) = (t - 2)^2 + 2(\frac{1}{4}(3 - t) - 1)^2
    = (t - 2)^2 + \frac{1}{8}(-t - 1)^2 \\
    f_1'(t) = 2(t-2) + \frac{1}{4}(-t -1) = \frac{9}{4} t - \frac{15}{4} \stackrel{!}{=} 0
    \Rightarrow t = \frac{15}{9} 
''')
g1_minx = 15/9
g1_miny = (3 - g1_minx)/4
g1_minf = (g1_minx - 2)**2 + 2*(g1_miny - 1)**2

st.write(r'Minimum auf dem Rand von $g_2$')
st.latex(r'''
    \partial g_2: h_2(t) = t \\
    f(t, h_2(t)) = f_2(t) = (t - 2)^2 + 2(t - 1)^2\\
    f_2'(t) = 2(t-2) + 4(t -1) = 6t - 8 \stackrel{!}{=} 0
    \Rightarrow t = \frac{4}{3} 
''')
g2_minx = 4/3
g2_miny = g2_minx
g2_minf = (g2_minx - 2)**2 + 2*(g2_miny - 1)**2

st.write(r'Wert am Schnittpunkt von $g_1$ und $g_2$')
st.latex(r'''
    h_1(t) = h_2(t) 
    \Rightarrow \frac{1}{4}(3 - t) = t
    \Leftrightarrow t = \frac{3}{5}
''')
g12_minx = 3/5
g12_miny = g12_minx
g12_minf = (g12_minx - 2)**2 + 2*(g12_miny - 1)**2

n = 100
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)

A = np.zeros((n,n))
A_ = np.zeros((n,n))
B = np.zeros((n,n))
C = np.zeros((n,n,))
g1 = np.zeros(n)
y1 = np.zeros(n)
g2 = np.zeros(n)
y2 = np.zeros(n)
for i in range(0,n):
    for j in range(0,n):
        A[i,j] = (x[j] - 2)**2 + 2*(y[i] - 1)**2

        A_[i,j] = A[i,j]
        B[i,j] = x[j] + 4*y[i] - 3
        if B[i,j] > 0:
            B[i,j] = None
            A_[i,j] = None

        C[i,j] = -x[j] + y[i]
        if C[i,j] > 0:
            C[i,j] = None
            A_[i,j] = None
     
    y1[i] = (3 - x[i])/4
    # g1[i] = (x[i] - 2)**2 + 2*(y1[i] - 1)**2
    g1[i] = (x[i] - 2)**2 + 1/8*(-x[i] - 1)**2

    y2[i] = x[i] 
    g2[i] = (x[i] - 2)**2 + 2*(y2[i] - 1)**2

# fig, ax = plt.subplots()
# ax.plot(x, g2)
# ax.plot(g2_minx, g2_minf, '.')
# st.pyplot(fig)

fig = go.Figure()
fig.add_trace(go.Surface(x=x, y=y, z=A, 
                         opacity=0.2, 
                         contours_z=dict(
                             show=True
                         ), 
                         showlegend=False,
                         showscale=False))
fig.add_trace(go.Scatter3d(x=np.array([f_minx]), y=np.array([f_miny]), z=np.array([f_minf]),
                mode='markers',
                showlegend=False,
                marker=dict(
                    color='black',
                    size=5
                )))

fig.add_trace(go.Surface(x=x, y=y, z=A_, 
                         showlegend=False,
                         showscale=False))

fig.add_trace(go.Scatter3d(x=x, y=y1, z=g1,
                mode='lines',
                showlegend=False,
                line=dict(
                    color='red',
                )))
fig.add_trace(go.Scatter3d(x=np.array([g1_minx]), y=np.array([g1_miny]), z=np.array([g1_minf]),
                mode='markers',
                showlegend=False,
                marker=dict(
                    color='red',
                    size=5
                )))


fig.add_trace(go.Scatter3d(x=x, y=y2, z=g2,
                mode='lines',
                showlegend=False,
                line=dict(
                    color='blue',
                )))
fig.add_trace(go.Scatter3d(x=np.array([g2_minx]), y=np.array([g2_miny]), z=np.array([g2_minf]),
                mode='markers',
                showlegend=False,
                marker=dict(
                    color='blue',
                    size=5
                )))

fig.add_trace(go.Scatter3d(x=np.array([g12_minx]), y=np.array([g12_miny]), z=np.array([g12_minf]),
                mode='markers',
                showlegend=False,
                marker=dict(
                    color='gray',
                    size=5
                )))

fig.update_layout(dict(
    width=700,
    height=700
))

st.plotly_chart(fig)


st.write(r'Lagrangefunktion $\mathcal{L}$')
st.latex(r'''
    \mathcal{L}(x,y,\lambda_1, \lambda_2) = f(x,y) + \lambda_1 g_1(x,y) + \lambda_2 g_2(x,y)
''')