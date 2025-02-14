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

    st.write("...")

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

    return A, n, m, parties

A, n, m_all, parties = get_data(data_dir)




# st.write(A, n, m)

# st.stop()

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



def plot_parliament(A, n, m, parties):



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
    fig, ax = plt.subplots()
    legend = []
    for i, party in enumerate(parties):
        label = None
        if party not in legend:
            legend.append(party)
            label = party

        ax.plot(A_d[i,0], A_d[i,1], '.', color=party_colors[party], alpha=0.4, label=label, markersize=10)

    # ax.plot(A_[:,0], A_[:,1], 'k.')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.set_xlabel(r'1. Hauptachse, $\tilde{a}_1$')
    ax.set_ylabel(r'2. Hauptachse, $\tilde{a}_2$')
    # ax.set_aspect('equal')
    # st.pyplot(fig)

    return A_d, fig, ax


def k_means(data, n_clusters, iterations=10, random_init=False, init_centroids=None, xaxis_plot_index=0, yaxis_plot_index=1):
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    dim = data.shape[0] # dimension
    n_points = data.shape[1] # number of data points

    if init_centroids is not None:
        centroids = init_centroids
    elif random_init:
        # Initialize centroids randomly
        centroids = np.random.rand(dim, n_clusters) * np.amax(data, axis=1).reshape(dim,1)

    else:
        # initialize centroids on a diagonal through the n-dimensional space
        centroids = np.zeros((dim, n_clusters))
        for i in range(0, n_clusters):
            centroids[:,i] = np.amin(data, axis=1) + (np.amax(data, axis=1) - np.amin(data, axis=1)) * (i+1)/(n_clusters+1)

    # centroids = np.zeros((dim, n_clusters))
    # for i in range(0, n_clusters):
    #     centroids[:,i] = np.ones(dim) * np.amax(data)*i/n_clusters + np.random.rand()

    fig, ax = plt.subplots()
    for j in range(0, iterations):
        
        new_centroids = np.zeros((dim, n_clusters))
        points_count_per_cluster = np.zeros(n_clusters)
        indices_per_cluster = []
        for k in range(0, n_clusters):
            indices_per_cluster.append(np.array([], dtype=int))


        for i in range(0, n_points):
            distances = np.zeros(n_clusters)
            for k in range(0, n_clusters):
                distances[k] = np.linalg.norm(data[:,i] - centroids[:,k])

            cluster_index = np.argmin(distances)
            points_count_per_cluster[cluster_index] += 1
            new_centroids[:,cluster_index] += data[:,i]

            indices_per_cluster[cluster_index] = np.append(indices_per_cluster[cluster_index], i)

            # centroids[i,:] = centroids[i,:] * np.amax(data[i,:])

        for k in range(0, n_clusters):

            points_per_cluster = data[:,indices_per_cluster[k]]
            if j == iterations-1:
                ax.plot(points_per_cluster[xaxis_plot_index,:], points_per_cluster[yaxis_plot_index,:], '.', color=color_list[k%len(color_list)], alpha=0.3)
            # centroids[:,k] = new_centroids[:,k] / points_count_per_cluster[k]
            old_centroid = np.copy(centroids[:,k])
            # centroids[:,k] = np.median(points_per_cluster, axis=1)
            # centroids[:,k] = np.quantile(points_per_cluster, 0.4, axis=1)
            centroids[:,k] = np.mean(points_per_cluster, axis=1)
            if j == iterations-1:
                ax.plot([old_centroid[xaxis_plot_index], centroids[xaxis_plot_index,k]], [old_centroid[yaxis_plot_index], centroids[yaxis_plot_index,k]], '--', color=color_list[k%len(color_list)])
                ax.plot(centroids[xaxis_plot_index,k], centroids[yaxis_plot_index,k], 'X', color='white')
                ax.plot(centroids[xaxis_plot_index,k], centroids[yaxis_plot_index,k], 'x', label=f'Cluster {k+1}', color=color_list[k%len(color_list)])

    points_near_center = []
    all_distances_to_center = []
    for k in range(0, n_clusters):
        points_per_cluster = data[:,indices_per_cluster[k]]
        distances_to_center = np.linalg.norm(points_per_cluster - centroids[:,k].reshape(dim,1), axis=0)
        all_distances_to_center.append(distances_to_center)

        if points_per_cluster.size == 0:
            continue    
        below_threshold = np.where(distances_to_center < np.quantile(distances_to_center, 1.0))
        # ax.plot(points_per_cluster[xaxis_plot_index,below_threshold], points_per_cluster[yaxis_plot_index,below_threshold], '.', color=color_list[k%len(color_list)], alpha=0.5)
        points_near_center.append(np.copy(indices_per_cluster[k][below_threshold]))

    ax.legend()
    st.pyplot(fig)
    st.caption('k-means Visualisierung der Clusterbildung (es werden jeweils nur 2 Dimensionen für den Plot verwendet)')

    return points_near_center, indices_per_cluster, all_distances_to_center

m = st.slider("m", 0, m_all, 62)
A_end = A[:,-m:]
A_start = A[:,:m]

A_start_d, fig, ax = plot_parliament(A_start, n, m, parties)
st.pyplot(fig)

A_min = np.amin(A_start_d, axis=0)
A_max = np.amax(A_start_d, axis=0)
init_centroids = np.array([[A_max[0], 0, A_min[0], 0],
                           [0, A_max[1], 0, A_min[0]]], dtype=float)

k_means(A_start_d.T, 4, init_centroids=init_centroids, iterations=10)

A_end_d, fig, ax = plot_parliament(A_end, n, m, parties)
st.pyplot(fig)

A_min = np.amin(A_start_d, axis=0)
A_max = np.amax(A_start_d, axis=0)
init_centroids = np.array([[A_max[0], 0, A_min[0], 0],
                           [0, A_max[1], 0, A_min[0]]], dtype=float)

k_means(A_end_d.T, 4, init_centroids=init_centroids, iterations=10)








