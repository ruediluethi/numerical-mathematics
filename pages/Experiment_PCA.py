import os
import random
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data_dir = os.path.join('data', 'bundestag')

party_colors = {
    'SPD': '#e3010f',
    'CDU/CSU': '#191919',
    'FDP': '#ffe300',
    'DIE LINKE.': '#e60e98',
    'BÜ90/GR': '#65a129',
    'Fraktionslos': '#aaaaaa',
    'AfD': '#00acd3',
}

st.header('Clustering der namentlichen Abstimmungen des 20. Bundestages mittels Principal Component Analysis (PCA)')
st.write('Datenquelle: https://www.bundestag.de/parlament/plenum/abstimmung/liste')

@st.cache_data
def get_data(data_dir):

    file_list = os.listdir(data_dir)
    n = len(file_list) # dimension / amount of votes

    p_names = None

    for v_index, file in enumerate(file_list):
        raw_file = os.path.join(data_dir, file)
        df = pd.read_excel(raw_file)
        df.describe()

        # st.write(df.groupby(['Fraktion/Gruppe']).mean())
        # exit()

        if p_names is None:
            p_names = df['Bezeichnung'].tolist()
            parties = df['Fraktion/Gruppe'].tolist()
            m = len(p_names) # amount of politicians
            A = np.zeros((m, n))

        for index, row in df.iterrows():
            if row['Bezeichnung'] in p_names:
                p_index = p_names.index(row['Bezeichnung'])
                A[p_index, v_index] = (row['ja'] - row['nein']) #+random.uniform(-0.1, 0.1)

        v_i = A[:, v_index]
        A[:, v_index] = (v_i - np.mean(v_i)) / np.std(v_i)

    return A, m, n, parties


A, m, n, parties = get_data(data_dir)

# calc covariance matrix
ATA = A.T @ A

# get eigenvalues and eigenvectors from covariance matrix
lambdas, V = np.linalg.eig(ATA)


# in comparision the singular values decomposition

# Sigma = np.zeros((m,n))
# Sigma[:n,:n] = np.diag(np.sqrt(lambdas))

# U = np.zeros((m,m))
# for i in range(0,n):
#     U[:,i] = 1/Sigma[i,i] * A @ V[:,i]

# for k in range(n,m):
#     e_k = np.zeros(m)
#     e_k[k] = 1
#     u_k_ = e_k
#     for j in range(0,k):
#         u_k_ = u_k_ - U[:,j].T @ e_k * U[:,j]
    
#     U[:,k] = u_k_/np.linalg.norm(u_k_)

# st.write(np.sum((U @ Sigma @ V.T) - A))

# same with the built in function from numpy
# U, S, Vh = np.linalg.svd(A)
# st.write(U, S, Vh)

# use only two eigenvalues and eigenvectors for 2D plot
n_ = 2
Sigma_ = np.zeros((m,n_))
Sigma_[:n_,:n_] = np.diag(np.sqrt(lambdas[:n_]))
V_ = V[:,:n_]

# project the data onto the new 2D basis
A_ = np.zeros((m,n_))
for i in range(0,n_):
    A_[:,i] = A @ V_[:,i]

# plot the data
fig, ax = plt.subplots()
legend = []
for i, party in enumerate(parties):
    label = None
    if party not in legend:
        legend.append(party)
        label = party

    ax.plot(A_[i,0], A_[i,1], '.', color=party_colors[party], alpha=0.4, label=label)

# ax.plot(A_[:,0], A_[:,1], 'k.')
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
st.pyplot(fig)
