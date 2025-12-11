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
    'Die Linke': '#e60e98',
    'BÜ90/GR': '#65a129',
    'BSW': '#7d254f',
    'Fraktionslos': '#aaaaaa',
    'AfD': '#00acd3',
}

st.header('Clustering der namentlichen Abstimmungen des 20. Bundestages mittels Principal Component Analysis (PCA)')
st.write('''
    Datenquelle: https://www.bundestag.de/parlament/plenum/abstimmung/liste     
    Es wurden alle namentlichen Abstimmungen des 20. Bundestages bis zum 5. Juli 2024 verwendet.
''')
# https://de.wikipedia.org/wiki/Bruch_der_Ampelkoalition_in_Deutschland_2024

# fetch and cache data
@st.cache_data
def get_data(data_dir):

    file_list = os.listdir(data_dir)
    m = len(file_list) # dimension / amount of votes

    p_names = None

    file_list = sorted(file_list)

    read_progress = st.progress(0.0)
    for v_index, file in enumerate(file_list):
        read_progress.progress(v_index/len(file_list), text=file)
        raw_file = os.path.join(data_dir, file)
        if ".DS_Store" in raw_file:
            continue
        df = pd.read_excel(raw_file)
        df.describe()

        # st.write(df.groupby(['Fraktion/Gruppe']).mean())
        # exit()

        if p_names is None:
            p_names = df['Bezeichnung'].tolist()
            parties = df['Fraktion/Gruppe'].tolist()
            n = len(p_names) # amount of politicians
            A = np.zeros((n, m))

        for index, row in df.iterrows():
            if row['Bezeichnung'] in p_names:
                p_index = p_names.index(row['Bezeichnung'])
                A[p_index, v_index] = (row['ja'] - row['nein']) #+random.uniform(-0.1, 0.1)

        v_i = A[:, v_index]
        A[:, v_index] = (v_i - np.mean(v_i)) / np.std(v_i)

    read_progress.empty()

    return A, n, m, parties, p_names, file_list

df_data_filepath = os.path.join(data_dir, 'df_data.csv')
df_A_filepath = os.path.join(data_dir, 'A.csv')
df_file_list_filepath = os.path.join(data_dir, 'file_list.csv')

if os.path.exists(df_data_filepath) and os.path.exists(df_A_filepath) and os.path.exists(df_file_list_filepath):
    df_data = pd.read_csv(df_data_filepath)
    df_A = pd.read_csv(df_A_filepath)
    df_file_list = pd.read_csv(df_file_list_filepath)

    parties = df_data['parties'].to_numpy()
    p_names = df_data['p_names'].to_numpy()

    file_list = df_file_list.to_numpy()[:,1:].flatten()

    A = df_A.to_numpy()[:,1:] # remove index col at beginning
    n = A.shape[0]
    m_all = A.shape[1]

else:
    A, n, m_all, parties, p_names, file_list = get_data(data_dir)



    df_data = pd.DataFrame(columns=['parties', 'p_names', 'file_list'])
    df_A = pd.DataFrame(A)
    df_file_list = pd.DataFrame(file_list)

    df_data['parties'] = parties
    df_data['p_names'] = p_names

    df_data.to_csv(df_data_filepath)
    df_A.to_csv(df_A_filepath)
    df_file_list.to_csv(df_file_list_filepath)

st.write(r'''
    Sei $X_j$ ein Zufallsvektor welcher die $j$-te namentliche Abstimmung repräsentiert.
    Dabei ist eine Komponenten $x_i \in \left[-1, 0, 1\right]$ die jeweilige Stimme des $i$-ten Politikers.
    Der Wert $x_i = 1$ bedeutet eine Ja, $x_i = -1$ eine Nein Stimme und $x_i = 0$ eine Enthaltung.

    Die Zufallsvektoren $X_j$ werden mit $\frac{X_j - \mathbb{E}(X_j)}{\text{Var}(X_j)}$ normalisiert so dass $\mathbb{E}(X_j) = 0$ und $\text{Var}(X_j) = 1$ gilt
    und in der Matrix $A \in \mathbb{R}^{n \times m}$ als Spalten gespeichert.
    Dabei sind $n$ die Anzahl der Politiker und $m$ die Anzahl der Abstimmungen.

    Das Matrixprodukt $A^\top A$ wird berechnet was der Korrelationsmatrix entspricht:
''')

st.latex(r'''
    \text{Corr}(X_j,X_k)
    = \frac{\mathbb{E} \left( 
    \left(X_j - \overbrace{\mathbb{E}(X_j)}^{=0} \right) \left(X_k - \overbrace{\mathbb{E}(X_k)}^{=0} \right)\right)}
    {\sqrt{\underbrace{\text{Var}(X_j)}_{=1} \underbrace{\text{Var}(X_k)}_{=1}}}
    = \mathbb{E} \left( X_j X_k \right)
    = \frac{1}{n} a_j^\top a_k
''')

st.write(r'''
    Von $A^\top A$ werden die Eigenwerte $\lambda_j$ und Eigenvektoren $v_j$ berechnet.
    Dann werden die Zeilenektoren $a_i$ von $A$ (alle Abstimmungsresultate eines jeweiligen Politikers) auf die $d = 2$ ersten Eigenvektoren/Hauptachsen projiziert:
''')
st.latex(r'''
    \alpha_i = a_i^\top v_i
    \quad \Leftrightarrow \quad
    \tilde{a}_i = A v_i
    \left(\begin{array}{c}
        \alpha_1 \\
        \vdots \\
        \alpha_i\\
        \vdots \\
        \alpha_n
    \end{array}\right) = A v_i
''')



def plot_parliament(A, parties, ax):

    n = A.shape[0]


    # calc covariance matrix
    ATA = 1/n * A.T @ A

    # st.write(n,m)
    # st.write(ATA.shape)

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
    d = 2
    Sigma_ = np.zeros((n,d))
    Sigma_[:d,:d] = np.diag(np.sqrt(lambdas[:d]))

    # project the data onto the new 2D basis
    A_d = np.zeros((n,d))
    for i in range(0,d):
        A_d[:,i] = A @ V[:,i]

    # plot the data
    legend = []
    for i, party in enumerate(parties):
        label = None
        if party not in legend:
            legend.append(party)
            label = party

        ax.plot(A_d[i,0], A_d[i,1], '.', color=party_colors[party], alpha=0.4, label=label, markersize=10)

    # ax.plot(A_[:,0], A_[:,1], 'k.')
    # ax.set_aspect('equal')
    # st.pyplot(fig)

    return A_d


fig, ax = plt.subplots()
A_all_d = plot_parliament(A, parties, ax)

ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
ax.set_xlabel(r'1. Hauptachse, $\tilde{a}_1$')
ax.set_ylabel(r'2. Hauptachse, $\tilde{a}_2$')
ax.set_title('Projektion über alle Abstimmungen')

st.pyplot(fig)


m = st.slider("m", 0, m_all, m_all-62)
A_start = A[:,:m]
A_end = A[:,m:]


fig, ax = plt.subplots(1,2, figsize=(10,5))

A_start_d = plot_parliament(A_start, parties, ax[0])
ax[0].set_title(f'von {file_list[0]}\nbis {file_list[m]}')

A_end_d = plot_parliament(A_end, parties, ax[1])
ax[1].set_title(f'von {file_list[m]}\nbis {file_list[-1]}')

st.pyplot(fig)



