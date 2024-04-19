import math
import os
import streamlit as st
import numpy as np
import pandas as pd

# from scipy import optimize
from PIL import Image

import matplotlib.pyplot as plt

st.title('Singulärwertzerlegung von Bildern')




# image = Image.open(os.path.join('data', 'images', 'butterfly256.png'))
image = Image.open(os.path.join('data', 'images', 'caterpillar256.png'))

data = np.asarray(image)

R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]

# R = np.tril(R) + np.tril(R).T - np.diag(np.diagonal(R))
# G = np.tril(G) + np.tril(G).T - np.diag(np.diagonal(G))
# B = np.tril(B) + np.tril(B).T - np.diag(np.diagonal(B))

n = np.shape(R)[0]

colored_image = st.container()
amount = st.slider('Anzahl Werte', 1, n, 15, 1)

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



# p = st.slider('Fade', 0.0, 1.0, 0.5)
# t = st.slider('t', 0.0, 1.0, 0.0, 0.01)
# n = 200
# for count in range(0, n):

# st.subheader(count)

# t = count/n
# amount = t*t*t * 200 +1
amount_prev = math.floor(amount)
amount_next = math.ceil(amount)
p = amount - amount_prev

# st.write(amount_prev, p, amount_next)
fig, axs = plt.subplots(3, 2)



R_reduced = lerp(reduce_matrix(R, amount_prev), reduce_matrix(R, amount_next), p)
axs[0][0].imshow(R, cmap='Greys')
axs[0][0].axis('off')
axs[0][1].imshow(R_reduced, cmap='Greys')
axs[0][1].axis('off')

G_reduced = lerp(reduce_matrix(G, amount_prev), reduce_matrix(G, amount_next), p)
axs[1][0].imshow(G, cmap='Greys')
axs[1][0].axis('off')
axs[1][1].imshow(G_reduced, cmap='Greys')
axs[1][1].axis('off')

B_reduced = lerp(reduce_matrix(B, amount_prev), reduce_matrix(B, amount_next), p)
axs[2][0].imshow(B, cmap='Greys')
axs[2][0].axis('off')
axs[2][1].imshow(B_reduced, cmap='Greys')
axs[2][1].axis('off')

st.pyplot(fig)

col1, col2 = colored_image.columns(2)

original = np.dstack((R,G,B))
image_original = Image.fromarray(original)
col1.image(image_original)


output = np.dstack((R_reduced,G_reduced,B_reduced))
image_output = Image.fromarray(output)
col2.image(image_output)

# image_output.save('output/test'+f"{count:03d}"+'.png')