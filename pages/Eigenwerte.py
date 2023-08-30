import math
import os
import streamlit as st
import numpy as np
import pandas as pd

from scipy import optimize
from PIL import Image

import matplotlib.pyplot as plt

st.title('Eigenwerte')




image = Image.open(os.path.join('data', 'images', 'caterpillar1024.png'))

data = np.asarray(image)

R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]

# R = np.tril(R) + np.tril(R).T - np.diag(np.diagonal(R))
# G = np.tril(G) + np.tril(G).T - np.diag(np.diagonal(G))
# B = np.tril(B) + np.tril(B).T - np.diag(np.diagonal(B))

n = np.shape(R)[0]


with st.sidebar:
    amount_of_singvalues = st.slider('Anzahl Werte', 1, n, 15, 1)

@st.cache_data
def reduce_matrix(A, amount):
    # D, P = np.linalg.eig(A)
    U, S, Vh = np.linalg.svd(A)

    # D_reduced = np.zeros((D.size, D.size))
    S_reduced = np.zeros((S.size, S.size))
    for i in range(0, amount):
        # D_reduced[i][i] = D[i]
        S_reduced[i][i] = S[i]

    # A_out = P @ D_reduced @ P.T
    A_out = U @ S_reduced @ Vh

    A_out = np.maximum(A_out, np.zeros(S.size))
    A_out = np.minimum(A_out, np.ones(S.size)*255)

    return A_out.astype(np.uint8)

def lerp(A, B, p):
    C = np.zeros(np.shape(A))
    for i in range(0, np.shape(A)[0]):
        for j in range(0, np.shape(A)[1]):
            C[i][j] = A[i][j] * (1-p) + B[i][j] * p
    
    return C.astype(np.uint8)

p = st.slider('Fade', 0.0, 1.0, 0.5)

t = st.slider('t', 0.0, 1.0, 0.0, 0.01)

n = 200
for count in range(0, n):

    st.subheader(count)

    t = count/n
    amount = t*t*t * 200 +1
    amount_prev = math.floor(amount)
    amount_next = math.ceil(amount)
    p = amount - amount_prev

    st.write(amount_prev, p, amount_next)

    R_reduced = lerp(reduce_matrix(R, amount_prev), reduce_matrix(R, amount_next), p)
    # fig, [ax1, ax2] = plt.subplots(1, 2)
    # ax1.imshow(R)
    # ax2.imshow(R_reduced)
    # st.pyplot(fig)

    G_reduced = lerp(reduce_matrix(G, amount_prev), reduce_matrix(G, amount_next), p)
    # fig, [ax1, ax2] = plt.subplots(1, 2)
    # ax1.imshow(G)
    # ax2.imshow(G_reduced)
    # st.pyplot(fig)

    B_reduced = lerp(reduce_matrix(B, amount_prev), reduce_matrix(B, amount_next), p)
    # fig, [ax1, ax2] = plt.subplots(1, 2)
    # ax1.imshow(B)
    # ax2.imshow(B_reduced)
    # st.pyplot(fig)

    output = np.dstack((R_reduced,G_reduced,B_reduced))
    image_output = Image.fromarray(output)
    st.image(image_output)

    # original = np.dstack((R,G,B))
    # image_original = Image.fromarray(original)
    # st.image(image_original)

    image_output.save('output/test'+f"{count:03d}"+'.png')