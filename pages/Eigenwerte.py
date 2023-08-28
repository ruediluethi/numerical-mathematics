import math
import os
import streamlit as st
import numpy as np
import pandas as pd

from scipy import optimize
from PIL import Image

import matplotlib.pyplot as plt

st.title('Eigenwerte')




image = Image.open(os.path.join('data', 'images', 'clouds512.png'))

data = np.asarray(image)

R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]

n = np.shape(R)[0]

with st.sidebar:
    amount_of_singvalues = st.slider('Anzahl Werte', 1, n, n, 1)

def reduce_matrix(A, amount):
    U, S, Vh = np.linalg.svd(A)

    S_reduced = np.zeros((S.size, S.size))
    for i in range(0, amount_of_singvalues):
        S_reduced[i][i] = S[i]

    A_out = U @ S_reduced @ Vh

    A_out = np.maximum(A_out, np.zeros(S.size))
    A_out = np.minimum(A_out, np.ones(S.size)*255)

    return A_out.astype(np.uint8)

R_reduced = reduce_matrix(R, amount_of_singvalues)
fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.imshow(R)
ax2.imshow(R_reduced)
st.pyplot(fig)

G_reduced = reduce_matrix(G, amount_of_singvalues)
fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.imshow(G)
ax2.imshow(G_reduced)
st.pyplot(fig)

B_reduced = reduce_matrix(B, amount_of_singvalues)
fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.imshow(B)
ax2.imshow(B_reduced)
st.pyplot(fig)

output = np.dstack((R_reduced,G_reduced,B_reduced))
image_output = Image.fromarray(output)

st.image(image_output)
st.image(image)
