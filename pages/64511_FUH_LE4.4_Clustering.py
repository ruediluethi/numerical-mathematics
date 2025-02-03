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

st.title('Lineare Diskriminanzfunktion')


st.write(r'''
    Sei $x_i = (1, {x_i}_1, ..., {x_i}_D)^\intercal \in \mathbb{R}^{D+1}$ ein Vektor, welcher die Merkmale eines Datenobjekts in der Dimension $D$ repräsentiert
    und sei $y_i \in \{1, -1\}$ die zugehörige Klasse des Datenobjekts.
    So ist eine lineare Abbildung gesucht, welche die Merkmale $x_i$ auf ihre Dimension $y_i$ abbildet.
''')



st.write('Datenquelle: https://www.kaggle.com/datasets/rtatman/lego-database')

df_themes = pd.read_csv('data/lego/themes.csv')
# df_themes = df_themes.set_index('id')
# st.write(df_themes)
df_themes['root'] = np.nan

theme_ids = df_themes['id'].to_numpy(dtype=int)




for i, row in df_themes.iterrows():

    parent_id = row['parent_id']
    if pd.isna(parent_id):
        df_themes.at[i, 'root'] = row['id']
        continue

    while not pd.isna(parent_id):
        parent = df_themes[df_themes['id'] == parent_id].iloc[0]
        parent_id = parent['parent_id']

    df_themes.at[i, 'root'] = parent['id']


st.write(df_themes)

themes_root = df_themes[df_themes['parent_id'].isnull()]
themes_root['count'] = 0
st.write(themes_root)


df_sets = pd.read_csv('data/lego/sets.csv')
st.write(df_sets)
df_sets_count = pd.DataFrame(df_sets.groupby('theme_id')['num_parts'].count().sort_values(ascending=False)).reset_index()

df_themes = df_themes.rename(columns={'id': 'theme_id'})

st.subheader('themes')
st.write(df_themes)
st.subheader('sets count')
st.write(df_sets_count)

st.write(pd.merge(df_sets_count, df_themes, on='theme_id'))

for i, row in df_sets_count.iterrows():
    theme = df_themes[df_themes['theme_id'] == row['theme_id']].iloc[0]
    # st.write(theme)
    root_index = themes_root[themes_root['id'] == theme['root']].index
    themes_root.loc[root_index, 'count'] += row['num_parts']
    # st.write(themes_root.loc[root_index])
    # st.write('...')

st.subheader('root themes')
st.write(themes_root.sort_values('count', ascending=False))

# root_sets_count = pd.DataFrame(df_sets_count.groupby('root')['num_parts'].count().sort_values(ascending=False)).join(df_themes)
# st.write(root_sets_count)

# for i, row in df_sets_count.iterrows():
#     parent_root = themes_root[themes_root['id'] == i]
#     if parent.shape[0] == 0:
#         # df_sets_count = df_sets_count.drop(i)
#         st.write(row, parent)





df_inventories = pd.read_csv('data/lego/inventories.csv')
df_parts = pd.read_csv('data/lego/parts.csv')
df_inv_parts = pd.read_csv('data/lego/inventory_parts.csv')

df_color_list = pd.read_csv('data/lego/colors.csv')
df_color_list = df_color_list.rename(columns={'id': 'color_id'})
st.write(df_color_list)

def w_fun(L, y_cap_scale=1):
    return np.clip((np.sin(L * np.pi*2 - np.pi/2)/2 + 0.5) * y_cap_scale, 0.0, 1.0)

#color_groups = [[], [], [], [], [], [], []]
color_groups = [[], [], [], 
                [], [], [], 
                [], [], []]

fig, ax = plt.subplots()
for i, row in df_color_list.iterrows():
    r, g, b = mcolors.hex2color('#'+row['rgb'])
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    #ax.plot([h], [s**2*w_fun(l)], 'o', color=(r, g, b))

    # no color
    if row['name'] == '[No Color]' or row['name'] == 'Unknown':
        st.write(row)

    # transparent
    elif row['is_trans'] == 't':
        color_groups[8].append(row['name'])
        ax.plot([h], [l], '.', color=(r, g, b), markersize=10, alpha=0.2)
        continue
    
    else:
        ax.plot([h], [l], '.', color=(r, g, b), markersize=10)

        # black
        if l < 0.15:
            ax.plot([h], [l], 'wx')
            color_groups[0].append(row['name'])

        # white
        elif l > 0.9:
            #ax.plot([h], [l], 'X', color=(r, g, b), markersize=10)
            ax.plot([h], [l], 'kx')
            color_groups[1].append(row['name'])

        # gray
        elif s < 0.18:
            ax.plot([h], [l], 'k+')
            color_groups[2].append(row['name'])

        # colored
        else:
            # ax.plot([h], [l], '.', color=(r, g, b), markersize=20)
            # color_groups[3].append(row['name'])

            #red
            if h < 0.05:
                ax.plot([h], [l], '.', color=(r, g, b), markersize=10)
                color_groups[3].append(row['name'])

            # yellow
            elif h < 0.17:
                ax.plot([h], [l], '.', color=(r, g, b), markersize=10)
                color_groups[4].append(row['name'])

            # purple
            elif h > 0.75:
                ax.plot([h], [l], '.', color=(r, g, b), markersize=10)
                color_groups[5].append(row['name'])

            # green
            elif h < 0.5:
                ax.plot([h], [l], '.', color=(r, g, b), markersize=10)
                color_groups[6].append(row['name'])

            # blue
            else:
                ax.plot([h], [l], '.', color=(r, g, b), markersize=20)
                color_groups[7].append(row['name'])
            # elif h < 0.1:
            #     ax.plot([h], [l], '.', color=(r, g, b), markersize=20)
    

st.pyplot(fig)

color_list = df_color_list['name'].to_list()
st.write(len(color_list))

@st.cache_data
def get_X(root_index, min_parts=100):

    all_set_nums = []
    all_set_names = []

    # color_list = gray_colors + red_colors
    
    X = np.ones((0, len(color_list)))

    # st.write(df_themes[df_themes['name'].str.contains(theme_name, case=False)])
    # for theme_id, row in df_themes[df_themes['name'].str.contains(theme_name, case=False)].iterrows():
    for i, row in df_themes[(df_themes['root'] == root_index) | (df_themes['theme_id'] == root_index)].iterrows():
        theme_id = row['theme_id']
        
        # st.subheader(theme_id)
        df_theme_sets = df_sets[df_sets['theme_id'] == theme_id]
        df_theme_sets = pd.merge(df_theme_sets, df_inventories, on='set_num')
        # st.write(df_theme_sets)
        for i, row in df_theme_sets.iterrows():
            
            df_parts_set = df_inv_parts[df_inv_parts['inventory_id'] == row['id']]
            df_colors = pd.DataFrame(df_parts_set.groupby('color_id')['quantity'].sum()).reset_index()
            df_colors = pd.merge(df_colors, df_color_list, on='color_id')
            
            if df_colors.shape[0] > 0:
                parts_count = df_colors['quantity'].sum()
                # st.write(f"{row['set_num']}: {row['name']} (id: {row['id']}), {parts_count} parts")
                if parts_count < min_parts:# and parts_count < 100:
                    continue

                # st.write(df_colors)
                # st.write(df_colors)

                fig, ax = plt.subplots()
                for i, row_ in df_colors.iterrows():
                    r, g, b = mcolors.hex2color('#'+row_['rgb'])
                    h, l, s = colorsys.rgb_to_hls(r, g, b)
                    #ax.plot([h], [s**2*w_fun(l)], 'o', color=(r, g, b))

                    ax.plot([h], [l], '.', color=(r, g, b), markersize=1+row_['quantity']/parts_count*50)

                    

                # st.pyplot(fig)

                X = np.vstack((X, np.zeros((1, len(color_list)))))
                all_set_nums.append(row['set_num'])
                all_set_names.append(row['name'])

                for i, row_col in df_colors.iterrows():
                    # if row_col['name'] not in color_list:
                    #     color_list.append(row_col['name'])
                    #     X = np.hstack((X, np.zeros((len(all_set_nums), 1))))

                    X[len(all_set_nums)-1, color_list.index(row_col['name'])] = row_col['quantity']

    parts_count = X.sum(axis=1)[:, np.newaxis]
    X = X / parts_count
                
    # st.write(X.shape)
    X_ = np.zeros((X.shape[0], len(color_groups)))
    for i in range(len(color_groups)):
        for j in range(len(color_groups[i])):
            X_[:,i] += X[:,color_list.index(color_groups[i][j])]

    # st.write(X_)

    # st.write(X)
    # st.write(X.shape)
    # st.write(color_list)

    # col_names = np.array(color_list)
    # col_count = np.sum(X, axis=0)
    # col_sort = np.argsort(col_count)[::-1]

    # st.write(pd.DataFrame({
    #     'color': col_names[col_sort],
    #     'count': col_count[col_sort]
    # }).head(20))

    return X_


#X_trains = get_X('Trains', min_parts=50)
# X_friends = get_X('Friends', min_parts=100)
#X_starwars = get_X('Star Wars', min_parts=200)

def PCA(A):
    n = A.shape[0]
    d = A.shape[1]
    ATA = 1/n * A.T @ A
    lambdas, V = np.linalg.eig(ATA)

    # st.write(lambdas)

    d = 2
    Sigma_ = np.zeros((n,d))
    Sigma_[:d,:d] = np.diag(np.sqrt(lambdas[:d]))

    # project the data onto the new 2D basis
    A_d = np.zeros((n,d))
    for i in range(0,d):
        A_d[:,i] = A @ V[:,i]

    return A_d


#set_names = ['City', 'Friends', 'Ninjago', 'Technic', 'Creator', 'Star Wars', 'Classic Space', 'Batman']
# set_names = ['Star Wars', 'Technic', 'Creator', 'Classic', 'Space', 'Train', 'Friends', 'Ninjago']
# set_names = ['Friends', 'Technic']

# set_names = st.multiselect('Sets', themes_root['name'].to_list(), default=['Friends', 'Technic'])

set_names = st.multiselect('Sets', themes_root.sort_values('count', ascending=False)['name'].to_list(), default=['Friends', 'Ninjago'])



X_list = []
for set_name in set_names:
    root = themes_root[themes_root['name'] == set_name].iloc[0]
    X_list.append(get_X(root['root'], min_parts=50))

# X_list.append(get_X(130.0, min_parts=50))
# set_names.append('Classic Space')

# X_list.append(get_X(571.0, min_parts=50))
# set_names.append('Legends of Chima')

# X_list.append(get_X(390.0, min_parts=50))
# set_names.append('Fabuland')

# root_themes = [494]

# X_list = []
# for root_index in root_themes:
#     X_list.append(get_X(root_index, min_parts=100))

# X = np.zeros((0, len(color_list)))
X = np.zeros((0, len(color_groups)))
for X_ in X_list:
    X = np.vstack((X, X_))

x_axis = st.selectbox('X-Achse', color_groups)
y_axis = st.selectbox('Y-Achse', color_groups)

fig, ax = plt.subplots()
k = 0
for i, X_ in enumerate(X_list):
    k_next = k + X_.shape[0]
    ax.plot(X[k:k_next,color_groups.index(x_axis)], 
            X[k:k_next,color_groups.index(y_axis)], '.', alpha=0.5, label=set_names[i])
    ax.set_xlabel(x_axis[0:3])
    ax.set_ylabel(y_axis[0:3])
    k = k_next

ax.legend()
st.pyplot(fig)

A = PCA(X)

fig, ax = plt.subplots()
k = 0
for i, X_ in enumerate(X_list):
    k_next = k + X_.shape[0]
    ax.plot(A[k:k_next,0], A[k:k_next,1], '.', alpha=0.5, label=set_names[i])
    k = k_next

ax.legend()
st.pyplot(fig)

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

def k_means(data, n_clusters, iterations=10, random_init=False, xaxis_plot_index=0, yaxis_plot_index=1):

    dim = data.shape[0] # dimension
    n_points = data.shape[1] # number of data points

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
            centroids[:,k] = np.mean(points_per_cluster, axis=1)
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

# st.write(A)

k_means(A.T, 2)