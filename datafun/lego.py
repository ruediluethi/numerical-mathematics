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

@st.cache_data
def load_lego_data(plot_colors=False):
    df_themes = pd.read_csv('data/lego/themes.csv')
    # df_themes = df_themes.set_index('id')
    # st.write(df_themes)
    df_themes['root'] = np.nan

    theme_ids = df_themes['id'].to_numpy(dtype=int)

    # search recursively for root theme
    for i, row in df_themes.iterrows():

        parent_id = row['parent_id']
        if pd.isna(parent_id):
            df_themes.at[i, 'root'] = row['id']
            continue

        while not pd.isna(parent_id):
            parent = df_themes[df_themes['id'] == parent_id].iloc[0]
            parent_id = parent['parent_id']

        df_themes.at[i, 'root'] = parent['id']


    # st.write(df_themes)

    themes_root = df_themes[df_themes['parent_id'].isnull()]
    themes_root['count'] = 0
    # st.write(themes_root)


    df_sets = pd.read_csv('data/lego/sets.csv')
    # group sets by theme and count sets in num_parts
    df_sets_count = pd.DataFrame(df_sets.groupby('theme_id')['num_parts'].count().sort_values(ascending=False)).reset_index()

    df_themes = df_themes.rename(columns={'id': 'theme_id'})

    # st.write(df_themes)
    # st.subheader('sets count')
    # st.write(df_sets_count)
    # st.write(pd.merge(df_sets_count, df_themes, on='theme_id'))

    # search for root theme and counts all sets of a root theme
    for i, row in df_sets_count.iterrows():
        theme = df_themes[df_themes['theme_id'] == row['theme_id']].iloc[0]
        # st.write(theme)
        root_index = themes_root[themes_root['id'] == theme['root']].index
        themes_root.loc[root_index, 'count'] += row['num_parts']
        # st.write(themes_root.loc[root_index])
        # st.write('...')

    with st.expander('root themes with sets count'):
        st.write(themes_root.sort_values('count', ascending=False))

    # root_sets_count = pd.DataFrame(df_sets_count.groupby('root')['num_parts'].count().sort_values(ascending=False)).join(df_themes)
    # st.write(root_sets_count)

    # for i, row in df_sets_count.iterrows():
    #     parent_root = themes_root[themes_root['id'] == i]
    #     if parent.shape[0] == 0:
    #         # df_sets_count = df_sets_count.drop(i)
    #         st.write(row, parent)

    # get part list for all sets
    df_inventories = pd.read_csv('data/lego/inventories.csv')
    df_parts = pd.read_csv('data/lego/parts.csv')
    df_inv_parts = pd.read_csv('data/lego/inventory_parts.csv')

    # get colors
    df_color_list = pd.read_csv('data/lego/colors.csv')
    df_color_list = df_color_list.rename(columns={'id': 'color_id'})
    # st.write(df_color_list)

    only_three = False
    # color_groups: list[list[str]] = []
    # if only_three:
    #     color_groups = [[], [], []]
    # else:
    color_groups: list[list[str]] = [[], [], [], [], [], [], [], [], []]

    if plot_colors:
        fig, ax = plt.subplots(figsize=(8,2))
        for i, row in df_color_list.iterrows():
            r, g, b = mcolors.hex2color('#'+row['rgb'])
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            #ax.plot([h], [s**2*w_fun(l)], 'o', color=(r, g, b))

            # no color
            if row['name'] == '[No Color]' or row['name'] == 'Unknown':
                # st.write(row)
                continue

            # transparent
            elif row['is_trans'] == 't':
                if not only_three:
                    color_groups[8].append(row['name'])
                ax.plot([h], [l], '.', color=(r, g, b), markersize=10, alpha=0.2)
                continue
            
            else:
                ax.plot([h], [l], '.', color=(r, g, b), markersize=10)

                if only_three:
                    if l < 0.2:
                        ax.plot([h], [l], 'wx')
                        color_groups[0].append(row['name'])    
                    elif s < 0.2:
                        ax.plot([h], [l], 'k+')
                        color_groups[1].append(row['name'])
                    else:
                        color_groups[2].append(row['name'])
                    continue

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
                        ax.plot([h], [l], '.', color=(r, g, b), markersize=10)
                        color_groups[7].append(row['name'])
                    # elif h < 0.1:
                    #     ax.plot([h], [l], '.', color=(r, g, b), markersize=20)
            
        # with st.expander('color groups (not used)'):
        ax.set_xlabel('Farbton')
        ax.set_ylabel('Helligkeit')
        st.pyplot(fig)
        st.caption('''
            Alle mögliche Farben der Lego-Steine geordnet nach Farbton und Helligkeit.
        ''')

    color_list = df_color_list['name'].to_list()
    # st.write(len(color_list))

    df_part_types = pd.read_csv('data/lego/part_categories.csv')
    part_types_ids = df_part_types['id'].to_list()
    part_types_names = df_part_types['name'].to_list()

    return [df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names]

def select_set_names(D, label="Wähle Themen", default_set_names=[]):
    df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D
    set_names_selection = st.multiselect(label, themes_root.sort_values('count', ascending=False)['name'].to_list(), default=default_set_names)
    return set_names_selection

def select_examples(D):
    default_set_names_A = []
    default_set_names_B = []

    selection = st.pills(
        "Examples",
        options=['Turtles vs. Princesses', 'Boys vs. Girls', 'Technic vs. Star Wars'],
        selection_mode="single",
    )

    if selection == 'Boys vs. Girls':
        # default_set_names_A = ['Friends', 'Freestyle']
        default_set_names_A = ['Friends']
        default_set_names_B = ['Ninjago', 'Bionicle']
    elif selection == 'Turtles vs. Princesses':
        default_set_names_A = ['Teenage Mutant Ninja Turtles']
        default_set_names_B = ['Disney Princess']
    elif selection == 'Technic vs. Star Wars':
        default_set_names_A = ['Technic']
        default_set_names_B = ['Star Wars']

    set_names_A = select_set_names(D, label='Gruppe A besteht aus den Lego-Sets folgender Themen', default_set_names=default_set_names_A)
    set_names_B = select_set_names(D, label='und Gruppe B aus diesen Themen', default_set_names=default_set_names_B)
    min_parts = st.slider('Minimale Anzahl Steine pro Set', min_value=0, max_value=300, value=50, step=1)

    return set_names_A, set_names_B, min_parts

@st.cache_data
def get_X_color(D, root_index, min_parts=100, as_percent=True):
    df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D

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

    if as_percent:
        parts_count = X.sum(axis=1)[:, np.newaxis]
        X = X / parts_count
                
    # st.write(X.shape)
    return X



def get_X_color_for_set_names(D, set_names_selection, min_parts=50):
    df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D
    X_list = []
    for set_name in set_names_selection:
        root = themes_root[themes_root['name'] == set_name].iloc[0]
        X_list.append(get_X_color(D, root['root'], min_parts=min_parts))
    X = np.zeros((0, len(color_list)))
    for X_ in X_list:
        X = np.vstack((X, X_))
    return X, X_list

def get_X_themes(D, root_index, min_parts=100, as_percent=True):
    df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D

    all_set_nums = []
    all_set_names = []

    X = np.ones((0, len(part_types_ids)))


    # st.write(df_themes[df_themes['name'].str.contains(theme_name, case=False)])
    # for theme_id, row in df_themes[df_themes['name'].str.contains(theme_name, case=False)].iterrows():
    #     # st.subheader(theme_id)
    #     df_theme_sets = df_sets[df_sets['theme_id'] == theme_id]
    #     df_theme_sets = pd.merge(df_theme_sets, df_inventories, on='set_num')
        # st.write(df_theme_sets)
    for i, row in df_themes[(df_themes['root'] == root_index) | (df_themes['theme_id'] == root_index)].iterrows():
        theme_id = row['theme_id']
        
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

    # parts_count = X.sum(axis=1)[:, np.newaxis]
    # X = X / parts_count

    # st.write(X)

    # with st.expander(f'{theme_name}'):
    #     cat_sums = np.sum(X, axis=0)
    #     cat_sort_indices = np.argsort(cat_sums)[::-1]
    #     st.table(pd.DataFrame({
    #                     'Part Type': np.array(part_types_names)[cat_sort_indices], 
    #                     'Quantity': cat_sums[cat_sort_indices]}).head(10))

    return X, all_set_names, all_set_nums

def get_X_theme_for_set_names(D, set_names_selection, min_parts=50):
    df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D
    X_list = []
    all_set_names = []
    all_set_nums = []
    for set_name in set_names_selection:
        # X_list.append(get_X_themes(D, set_name, min_parts=min_parts))
        root = themes_root[themes_root['name'] == set_name].iloc[0]
        X_, set_names, set_nums = get_X_themes(D, root['root'], min_parts=min_parts)
        X_list.append(X_)
        all_set_nums.extend(set_nums)
        all_set_names.extend(set_names)
    X = np.zeros((0, len(part_types_names)))
    for X_ in X_list:
        X = np.vstack((X, X_))
    return X, X_list, all_set_names, all_set_nums