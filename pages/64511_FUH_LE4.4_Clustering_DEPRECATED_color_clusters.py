import os
import streamlit as st
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import colorsys

from scipy.signal import find_peaks

import random
import pandas as pd

COLOR_WHEEL_NAMES = ['Rot', 'Orange', 'Gelb', 'Limette', 'Grün', 'Türkis', 'Cyan', 'Azurblau', 'Blau', 'Lila', 'Magenta', 'Purpur']
  

st.title('Clustering')

# get data
photoset_path = os.path.join('data', 'photoset')

img_files_list = []
for file in os.listdir(photoset_path):
    img_path = os.path.join(photoset_path, file)
    if os.path.isfile(img_path):
        img_files_list.append(img_path.replace('\\', '/'))



img_path = random.choice(img_files_list)

# img_path = 'data/photoset/79.png'
img_path = 'data/photoset/82.png'

image = Image.open(img_path)
data = np.asarray(image)

R = data[:,:,0].flatten()[::100]
G = data[:,:,1].flatten()[::100]
B = data[:,:,2].flatten()[::100]

n = R.size
st.write(n)
H = np.zeros(n)
L = np.zeros(n)
S = np.zeros(n)

fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1,2)
ax_img = fig.add_subplot(gs[0,0])
ax_F = fig.add_subplot(gs[0,1])

ax_img.imshow(image)
ax_img.set_title('Datenobjekt / Bild')

ax_F.set_title(r'Merkmalsraum $\mathbb{F}$')
ax_F.set_xlabel('Lightness / Helligkeit')
ax_F.set_ylabel('Saturation / Sättigung')

for i in range(n):
    H[i], L[i], S[i] = colorsys.rgb_to_hls(R[i]/255, G[i]/255, B[i]/255)
    
    # red, green, blue = colorsys.hls_to_rgb(H[i], 0.5, 1.0)
    # ax_F.scatter(L[i], S[i], color=(R[i]/255, G[i]/255, B[i]/255), edgecolor='#000000')
    # ax_F.scatter(L[i], S[i], color=(red, green, blue), edgecolor='none')

st.pyplot(fig)



def w_fun(L, y_cap_scale=1.5):
    return np.clip((np.sin(L * np.pi*2 - np.pi/2)/2 + 0.5) * y_cap_scale, 0.0, 1.0)

A = np.zeros((n,3))
A[:,0] = H
A[:,1] = L
A[:,2] = S

st.write(A)

ATA = 1/n * A.T @ A
lambdas, V = np.linalg.eig(ATA)

st.write(lambdas)

d = 2
Sigma_ = np.zeros((n,d))
Sigma_[:d,:d] = np.diag(np.sqrt(lambdas[:d]))

# project the data onto the new 2D basis
A_d = np.zeros((n,d))
for i in range(0,d):
    A_d[:,i] = A @ V[:,i]

w = (S**3*w_fun(L[i], y_cap_scale=1.0))
w = w/np.amax(w)

# plot the data
fig, ax = plt.subplots()
for i in range(n):

    ax.scatter(A_d[i,0], A_d[i,1], color=(R[i]/255, G[i]/255, B[i]/255), 
                edgecolor='none', s=w[i]*300)

st.pyplot(fig)

A = np.zeros((n,3))
A[:,0] = R
A[:,1] = G
A[:,2] = B

ATA = 1/n * A.T @ A
lambdas, V = np.linalg.eig(ATA)

st.write(lambdas)

d = 2
Sigma_ = np.zeros((n,d))
Sigma_[:d,:d] = np.diag(np.sqrt(lambdas[:d]))

# project the data onto the new 2D basis
A_d = np.zeros((n,d))
for i in range(0,d):
    A_d[:,i] = A @ V[:,i]

# plot the data
fig, ax = plt.subplots()
for i in range(n):

    ax.scatter(A_d[i,0], A_d[i,1], color=(R[i]/255, G[i]/255, B[i]/255), 
                edgecolor='none', s=w[i]*300)

st.pyplot(fig)

fig, ax = plt.subplots()
for i in range(n):

    ax.scatter(H[i], w[i], color=(R[i]/255, G[i]/255, B[i]/255), 
                edgecolor='none', s=w[i]*300)

st.pyplot(fig)

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

def k_means(data, n_clusters, weights, iterations=10, random_init=False, xaxis_plot_index=0, yaxis_plot_index=1):

    dim = data.shape[0] # dimension
    n_points = data.shape[1] # number of data points

    st.write(dim, n_points)

    if random_init:
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
            
            # centroids[:,k] = np.mean(points_per_cluster, axis=1)
            centroids[:,k] = np.average(points_per_cluster, axis=1, weights=weights[indices_per_cluster[k]])

            if j == iterations-1:
                ax.plot([old_centroid[xaxis_plot_index], centroids[xaxis_plot_index,k]], [old_centroid[yaxis_plot_index], centroids[yaxis_plot_index,k]], '--', color=color_list[k%len(color_list)])
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
        ax.plot(points_per_cluster[xaxis_plot_index,below_threshold], points_per_cluster[yaxis_plot_index,below_threshold], '.', color=color_list[k%len(color_list)], alpha=0.5)
        points_near_center.append(np.copy(indices_per_cluster[k][below_threshold]))

    ax.legend()
    st.pyplot(fig)
    st.caption('k-means Visualisierung der Clusterbildung (es werden jeweils nur 2 Dimensionen für den Plot verwendet)')

    return points_near_center, indices_per_cluster, all_distances_to_center

# points_near_center, indices_per_cluster, distances_to_center = k_means(A_d[:,0:2].T, 3, w, random_init=False)

# points_near_center, indices_per_cluster, distances_to_center = k_means(A.T, 3, random_init=False)