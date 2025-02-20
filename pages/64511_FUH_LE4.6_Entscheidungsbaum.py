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

st.title('Entscheidungsbaum')



st.write('Datenquelle: https://www.kaggle.com/datasets/rtatman/lego-database')

df_themes = pd.read_csv('data/lego/themes.csv')
df_themes = df_themes.set_index('id')

df_sets = pd.read_csv('data/lego/sets.csv')
df_sets_count = pd.DataFrame(df_sets.groupby('theme_id')['num_parts'].count().sort_values(ascending=False))
with st.expander('Verfügbare Lego-Themes'):
    st.write(df_sets_count.join(df_themes))

df_inventories = pd.read_csv('data/lego/inventories.csv')
df_parts = pd.read_csv('data/lego/parts.csv')
df_inv_parts = pd.read_csv('data/lego/inventory_parts.csv')
df_part_types = pd.read_csv('data/lego/part_categories.csv')
with st.expander('Verfügbare Lego-Steine-Typen'):
    st.write(df_part_types)

part_types_ids = df_part_types['id'].to_list()
part_types_names = df_part_types['name'].to_list()
# st.write(part_types_ids)

@st.cache_data
def get_X(theme_name, min_parts=100):

    # st.header(theme_name)

    all_set_nums = []
    all_set_names = []

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

    parts_count = X.sum(axis=1)[:, np.newaxis]
    X = X / parts_count
    # st.write(X)

    with st.expander(f'{theme_name}'):
        cat_sums = np.sum(X, axis=0)
        cat_sort_indices = np.argsort(cat_sums)[::-1]
        st.table(pd.DataFrame({
                        'Part Type': np.array(part_types_names)[cat_sort_indices], 
                        'Quantity': cat_sums[cat_sort_indices]}).head(10))

    return X

    X_ = np.zeros((X.shape[0], 5))
    X_[:,0] = X[:,part_types_names.index('Bricks')] + X[:,part_types_names.index('Bricks Sloped')] + X[:,part_types_names.index('Bricks Round and Cones')] + X[:,part_types_names.index('Bricks Curved')]
    X_[:,1] = X[:,part_types_names.index('Plates')] + X[:,part_types_names.index('Plates Special')] + X[:,part_types_names.index('Plates Round and Dishes')] + X[:,part_types_names.index('Plates Angled')]
    X_[:,2] = X[:,part_types_names.index('Tiles')]+X[:,part_types_names.index('Tiles Printed')]
    X_[:,3] = X[:,part_types_names.index('Wheels and Tyres')]
    X_[:,4] = X[:,part_types_names.index('Technic Pins')] + X[:,part_types_names.index('Technic Axles')] + X[:,part_types_names.index('Technic Beams')] + X[:,part_types_names.index('Technic Bushes')] + X[:,part_types_names.index('Technic Bricks')]
    
    # X_[:,3] = X[:,part_types_names.index('Minifigs')]+X[:,part_types_names.index('Minifig Accessories')]

    st.write(X_)

    
    return X_

                
set_names = st.multiselect('Lego Themes', df_themes['name'].to_list(), default=['Technic', 'Friends'])

X_list = []
for set_name in set_names:
    X_list.append(get_X(set_name, min_parts=100))

st.header('Informationsgewinn (ID3, C4.5)')

st.write(r'''
    Menge der Lego-Sets $T$ aus $c_1, ..., c_n$ verschiedenen Lego-Themes Klassen
    und den Wahrscheinlichkeiten $p_k$, dass ein Lego-Set aus dem Lego-Theme $c_k$ stammt.
''')
st.latex(r'''
   entropy(T) = -\sum_{k=1}^{n} p_k \log_2 p_k     
''')

X = np.zeros((0, len(part_types_ids)+1))
for i, X_ in enumerate(X_list):
    X_ = np.hstack((X_, np.ones((X_.shape[0], 1))*i))
    X = np.vstack((X, X_))

def entropy(p, do_print=False):
    entropy_T = 0
    for i, set_name in enumerate(set_names):
        # count = X[X[:,-1] == i].shape[0]
        count = p[p == i].shape[0]
        p_i = count / p.shape[0]
        entropy_T -= p_i * np.log2(p_i)
        if do_print:
            st.write(f'$c_{i+1}$: {set_name} mit $p_{i+1}$ = {count} / {p.shape[0]} = {p_i:.3f}')
    return entropy_T

entropy_T = entropy(X[:,-1], do_print=True)
st.write(f'$entropy(T)$ = {entropy_T:.3f}')

st.write(r'''
    Sei $A$ das Attribut (der Legeo-Stein-Typ) und seien $T_1, ..., T_m$ die Partitionen von $T$. 
''')

st.latex(r'''
    informationGain(T, A) = entropy(T) - \sum_{i = 1}^{m} \frac{|T_i|}{|T|} \cdot entropy(T_i)
''')

for part_type in part_types_names: #['Technic Bricks', 'Bricks']:
    i = part_types_names.index(part_type)
    
    st.subheader(f'{i}: {part_type}')
    x = X[:,i]
    # st.write(x)

    x_threshold = np.mean(x)

    T_1 = np.argwhere(x <= x_threshold).flatten()
    T_2 = np.argwhere(x > x_threshold).flatten()

    information_gain = entropy_T

    # st.write(X[T_1,-1].shape, X[T_2,-1].shape, X[:,-1].shape)
    st.write(T_1.size, T_2.size, x.size)
    if T_1.size == 0 or T_2.size == 0:
        continue

    # st.write(f'{part_type} <= {x_threshold}')
    entropy_T_1 = entropy(X[T_1,-1])
    information_gain -= T_1.size / x.size * entropy_T_1

    # st.write(f'{part_type} > {x_threshold}')
    entropy_T_2 = entropy(X[T_2,-1])
    information_gain -= T_2.size / x.size * entropy_T_2

    st.write(f'$informationGain(T, A)$ = {information_gain:.3f}')

    # st.write(x_median)
    # x_1 = x[x <= x_median]
    # x_2 = x[x > x_median]

    # st.write(x_1.size, x_2.size)
