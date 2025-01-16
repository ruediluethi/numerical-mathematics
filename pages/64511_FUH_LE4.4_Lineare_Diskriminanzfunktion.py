import os
import streamlit as st
import numpy as np
import numpy.linalg as linalg

from PIL import Image

import matplotlib.pyplot as plt
import random

from scipy.stats import norm
import colorsys
import pandas as pd


st.title('Lineare Diskriminanzfunktion')

df = pd.DataFrame({ 
    'age': [23, 17, 43, 68, 32],
    'max_speed': [180, 240, 246, 173, 110],
    'risk': [1, 1, 1, -1, -1]
})

st.write(df)

age = df['age'].to_numpy()
max_speed = df['max_speed'].to_numpy()
y = df['risk'].to_numpy()

n = y.size

st.subheader('Zweidimensional')

X = np.ones((n,3))
X[:,1] = age
X[:,2] = max_speed

st.write(X)

pseudo_inv = linalg.inv(X.T @ X) @ X.T

st.write(pseudo_inv)

w = pseudo_inv @ y



st.write(w)

st.write(y - X @ w)

fig, ax = plt.subplots()
ax.plot(age[y==1], max_speed[y==1], 'ro', label='high risk')
ax.plot(age[y==-1], max_speed[y==-1], 'go', label='low rist')

ax.plot(age, (-w[0]-w[1]*age)/w[2], 'b-', label='decision boundary')

st.pyplot(fig)


st.subheader('Eindimensional')

n = 100

n_high = int(n*0.25 + random.randint(0, int(n*0.5)))
n_low = n - n_high

st.write(n, n_high, n_low, n_high + n_low)

high = np.random.normal(loc=3, scale=0.5, size=n_high)
low = np.random.normal(loc=1, scale=0.5, size=n_low)

y = np.concatenate((np.ones(n_high), -np.ones(n_low)))
X = np.ones((n,2))
X[:,1] = np.concatenate((high, low))

pseudo_inv = linalg.inv(X.T @ X) @ X.T
w = pseudo_inv @ y

fig, ax = plt.subplots()
ax.plot(np.zeros(n_high), X[:,1][y==1], 'ro', label='high risk')
ax.plot(np.zeros(n_low), X[:,1][y==-1], 'g.', label='low risk')

ax.plot([-1, 1], np.ones(2) * -w[0]/w[1], 'b-', label='decision boundary')


st.pyplot(fig)