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
import plotly.express as px
import plotly.graph_objects as go

st.title('Statistische Methoden')



st.write('Datenquelle: https://www.kaggle.com/datasets/rtatman/lego-database')

df_themes = pd.read_csv('data/lego/themes.csv')
df_themes = df_themes.set_index('id')

df_sets = pd.read_csv('data/lego/sets.csv')
df_sets_count = pd.DataFrame(df_sets.groupby('theme_id')['num_parts'].count().sort_values(ascending=False))
st.write('LEGO Themes')
st.write(df_sets_count.join(df_themes))

df_inventories = pd.read_csv('data/lego/inventories.csv')
df_parts = pd.read_csv('data/lego/parts.csv')
df_inv_parts = pd.read_csv('data/lego/inventory_parts.csv')
df_part_types = pd.read_csv('data/lego/part_categories.csv')
st.write('Part Types')
st.write(df_part_types)

part_types_ids = df_part_types['id'].to_list()
part_types_names = df_part_types['name'].to_list()
# st.write(part_types_ids)

# df_color_list = pd.read_csv('data/lego/colors.csv')
# df_color_list = df_color_list.rename(columns={'id': 'color_id'})
# # st.write(df_color_list)

# gray_colors = df_color_list[(df_color_list['name'].str.contains('Black')) |
#                             (df_color_list['name'].str.contains('Gray')) |
#                             (df_color_list['name'].str.contains('Opaque'))
#                         ]['name'].to_list()
# red_colors = df_color_list[
#                             (df_color_list['name'].str.contains('Red')) | 
#                             (df_color_list['name'].str.contains('Purple')) | 
#                             (df_color_list['name'].str.contains('Pink')) |
#                             (df_color_list['name'].str.contains('Lavender')) |
#                             (df_color_list['name'].str.contains('Magenta')) |
#                             (df_color_list['name'].str.contains('Orange')) |
#                             (df_color_list['name'].str.contains('Yellow'))
#                         ]['name'].to_list()

# red_colors = df_color_list[
#                             (df_color_list['name'].str.contains('Blue')) | 
#                             (df_color_list['name'].str.contains('Azure'))
#                         ]['name'].to_list()

# green_colors = df_color_list[
#                             (df_color_list['name'].str.contains('Green')) | 
#                             (df_color_list['name'].str.contains('Lime'))
#                         ]['name'].to_list()


def get_X(theme_name, min_parts=100):

    st.header(theme_name)

    all_set_nums = []
    all_set_names = []

    # color_list = gray_colors + red_colors

    X = np.ones((0, len(part_types_ids)))


    # st.write(df_themes[df_themes['name'].str.contains(theme_name, case=False)])
    for theme_id, row in df_themes[df_themes['name'].str.contains(theme_name, case=False)].iterrows():
        # st.subheader(theme_id)
        df_theme_sets = df_sets[df_sets['theme_id'] == theme_id]
        df_theme_sets = pd.merge(df_theme_sets, df_inventories, on='set_num')
        # st.write(df_theme_sets)

        for i, row in df_theme_sets.iterrows():
            # st.write(row['name'])
            
            df_parts_set = df_inv_parts[df_inv_parts['inventory_id'] == row['id']]
            # st.write(df_parts_set)
            df_parts_set_merged = pd.merge(df_parts_set, df_parts, on='part_num')
            # st.write(df_parts_set_merged)

            if df_parts_set_merged.shape[0] > 0:
                if df_parts_set_merged['quantity'].sum() < min_parts:
                    continue

                X = np.vstack((X, np.zeros((1, len(part_types_ids)))))
                all_set_nums.append(row['set_num'])
                all_set_names.append(row['name'])

                for i, row_col in df_parts_set_merged.iterrows():
                    cat_id = int(row_col['part_cat_id'])-1
                    X[len(all_set_nums)-1, cat_id] += row_col['quantity']

                # part_sort_indices = np.argsort(X[len(all_set_nums)-1,:])[::-1]
                # st.table(pd.DataFrame({
                #     'Part Type': np.array(part_types_names)[part_sort_indices], 
                #     'Quantity': X[len(all_set_nums)-1, part_sort_indices]}).head())

    # st.write(X)

    with st.expander('All sets'):
        st.table(pd.DataFrame({
            'Set Num': all_set_nums, 
            'Set Name': all_set_names}))

    parts_count = X.sum(axis=1)[:, np.newaxis]
    X = X / parts_count

    col_means = X.mean(axis=0)
    X_centered = X - col_means

    # st.write(X.shape)
    # st.write(X)
    # st.write(all_set_names)
    # st.write(part_types_names)

    cov_X = X_centered.T @ X_centered
    # st.write(cov_X)
    # st.write(cov_X.shape)

    with st.expander('Full covariance matrix'):
        fig_cov, ax_cov = plt.subplots()
        ax_cov.imshow(cov_X)
        st.pyplot(fig_cov)

    X_ = np.zeros((X.shape[0], 5))
    X_[:,0] = X[:,part_types_names.index('Bricks')] + X[:,part_types_names.index('Bricks Sloped')] + X[:,part_types_names.index('Bricks Round and Cones')] + X[:,part_types_names.index('Bricks Curved')]
    X_[:,1] = X[:,part_types_names.index('Plates')] + X[:,part_types_names.index('Plates Special')] + X[:,part_types_names.index('Plates Round and Dishes')] + X[:,part_types_names.index('Plates Angled')]
    X_[:,2] = X[:,part_types_names.index('Tiles')]+X[:,part_types_names.index('Tiles Printed')]
    X_[:,3] = X[:,part_types_names.index('Wheels and Tyres')]
    X_[:,4] = X[:,part_types_names.index('Technic Pins')] + X[:,part_types_names.index('Technic Axles')] + X[:,part_types_names.index('Technic Beams')] + X[:,part_types_names.index('Technic Bushes')] + X[:,part_types_names.index('Technic Bricks')]

    col_means_ = X_.mean(axis=0)
    X_centered_ = X_ - col_means_

    # X_[:,3] = X[:,part_types_names.index('Minifigs')]+X[:,part_types_names.index('Minifig Accessories')]

    # st.write(X_)

    fig_cov, ax_cov = plt.subplots()
    ax_cov.imshow(X_centered_.T @ X_centered_)
    st.pyplot(fig_cov)

    cat_sums = np.sum(X, axis=0)
    cat_sort_indices = np.argsort(cat_sums)[::-1]
    with st.expander('Parts count overview'):
        st.table(pd.DataFrame({
                    'Part Type': np.array(part_types_names)[cat_sort_indices], 
                    'Quantity': cat_sums[cat_sort_indices]}).head(10))
    return X_centered_

                
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

set_names = ['Technic', 'Friends', 'Ninjago']

#set_names = ['Friends', 'City', 'Ninjago', 'Creator', 'Basic', 'Star Wars', 'Technic', 'Architecture']
# set_names = ['City', 'Friends', 'Ninjago', 'Technic', 'Creator', 'Star Wars', 'Classic Space', 'Batman']
# set_names = ['City', 'Friends', 'Ninjago', 'Creator', 'Star Wars', 'Classic Space', 'Batman']
# set_names = ['Friends', 'Star Wars', 'Creator']
#set_names = ['Creator', 'Friends', 'Ninjago', 'Star Wars']

X_list = []
for set_name in set_names:
    X_list.append(get_X(set_name, min_parts=50))


# X = np.zeros((0, len(part_types_ids)))
X = np.zeros((0, 5))
for X_ in X_list:
    X = np.vstack((X, X_))

A = PCA(X)

fig, ax = plt.subplots()
k = 0
for i, X_ in enumerate(X_list):
    k_next = k + X_.shape[0]
    ax.plot(A[k:k_next,0], A[k:k_next,1], '.', alpha=0.5, label=set_names[i])
    k = k_next

ax.legend()
st.pyplot(fig)