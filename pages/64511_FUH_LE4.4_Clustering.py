import os
import streamlit as st
import numpy as np
import numpy.linalg as linalg

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import random

from scipy.stats import norm
import colorsys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title('Clustering')

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

df = pd.read_csv(os.path.join('data', 'bundestag.csv'))

st.table(df.head())

parties = df['Partei'].to_list()
A = df[['1. Hauptachse', '2. Hauptachse', '3. Hauptachse']].to_numpy()

# st.write(A)

fig, ax = plt.subplots()
legend = []
for i, party in enumerate(parties):
    label = None
    if party not in legend:
        legend.append(party)
        label = party

    ax.plot(A[i,0], A[i,1], '.', color=party_colors[party], alpha=0.4, label=label, markersize=10)

ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
ax.set_xlabel(r'1. Hauptachse, $\tilde{a}_1$')
ax.set_ylabel(r'2. Hauptachse, $\tilde{a}_2$')

st.pyplot(fig)

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

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

A_min = np.amin(A, axis=0)
A_max = np.amax(A, axis=0)
init_centroids = np.array([[A_max[0], 0, A_min[0], 0],
                           [0, A_max[1], 0, A_min[0]]], dtype=float)

st.write(init_centroids)

points_near_center, indices_per_cluster, all_distances_to_center = k_means(A[:,0:2].T, 4, init_centroids=init_centroids, iterations=10)

fig, ax = plt.subplots()
for k, indices in enumerate(indices_per_cluster):
    distances = all_distances_to_center[k]
    fig_, ax_ = plt.subplots()
    for j, i in enumerate(indices):
        party = parties[i]
        ax.plot(A[i,0], A[i,1], '.', color=color_list[k], alpha=0.4, markersize=10)
        ax.plot(A[i,0], A[i,1], '.', color=party_colors[party], markersize=4)

        ax_.plot(j, distances[j], '.', color=party_colors[party], markersize=4)

    st.pyplot(fig_)

st.pyplot(fig)
