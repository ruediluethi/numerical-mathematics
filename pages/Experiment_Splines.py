import streamlit as st
import numpy as np
import math 
from matplotlib import pyplot as plt


import plotly.graph_objects as go

import pandas as pd

df = pd.read_csv('data/cnc_mill_tool_wear/experiment_01.csv')
st.write(df)


# Extract X, Y, Z actual positions
x = df['X1_ActualPosition']
y = df['Y1_ActualPosition']
z = df['Z1_ActualPosition']

# Create a 3D line plot
fig = go.Figure(data=[
	go.Scatter3d(
		x=x,
		y=y,
		z=z,
		mode='lines',
		line=dict(color='blue', width=2),
		name='Tool Path (X, Y, Z)'
	)
])

fig.update_layout(
	scene=dict(
		xaxis_title='X1_ActualPosition',
		yaxis_title='Y1_ActualPosition',
		zaxis_title='Z1_ActualPosition',
	),
	title='CNC Tool Path in 3D',
	margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig)

values = st.slider("Select a range of values", 0, x.size, (460, 610))

x_ = x[values[0]:values[1]]
y_ = y[values[0]:values[1]]

fig, ax = plt.subplots()
ax.plot(x_, y_, '.', alpha=0.5)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(x_)
st.pyplot(fig)


def bernstein_base(t, i, n):
    n = n-1
    return math.comb(n,i) * math.pow(t, i) * math.pow(1 - t, n - i)


grade = 5
res = 100
t = np.linspace(0.0, 1.0, res)
s = np.zeros(res)
fig, ax = plt.subplots()
for k in range(0, grade):
    b = np.array([bernstein_base(t_i, k, grade) for t_i in t])
    s = s + b
    ax.plot(t, b)
ax.plot(t, s)
st.pyplot(fig)
