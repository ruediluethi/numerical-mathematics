import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datafun.lego import get_X_color_for_set_names, get_X_theme_for_set_names, load_lego_data, select_examples, select_set_names

st.title('Lego Database')

st.write('Datenquelle: https://www.kaggle.com/datasets/rtatman/lego-database')



D = load_lego_data(plot_colors=True)
df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D

set_names_A, set_names_B, min_parts = select_examples(D)

X_color_A, X_color_list_A = get_X_color_for_set_names(D, set_names_A, min_parts=min_parts)
X_color_B, X_color_list_B = get_X_color_for_set_names(D, set_names_B, min_parts=min_parts)
# st.write(X_color_A.shape, X_color_B.shape)

X_theme_A, X_theme_list_A, set_names_A, set_nums_A = get_X_theme_for_set_names(D, set_names_A, min_parts=min_parts)
X_theme_B, X_theme_list_B, set_names_B, set_nums_B = get_X_theme_for_set_names(D, set_names_B, min_parts=min_parts)
# st.write(X_theme_A.shape, X_theme_B.shape)


plot_colors_A = ['red', 'purple', 'orange']
plot_colors_B = ['blue', 'green', 'cyan']


with st.expander('Plot mit der Anzahl Lego-Steine einer Farbe pro Achse'):
    x_axis = st.selectbox('X-Achse', color_list, index=color_list.index('Black'))
    y_axis = st.selectbox('Y-Achse', color_list, index=color_list.index('White'))

    fig, ax = plt.subplots()
    k = 0
    for i, X_ in enumerate(X_color_list_A):
        k_next = k + X_.shape[0]
        ax.plot(X_color_A[k:k_next,color_list.index(x_axis)], 
                X_color_A[k:k_next,color_list.index(y_axis)], '.', alpha=0.5, label=set_names_A[i], color=plot_colors_A[i%len(plot_colors_A)])
        k = k_next
    k = 0
    for i, X_ in enumerate(X_color_list_B):
        k_next = k + X_.shape[0]
        ax.plot(X_color_B[k:k_next,color_list.index(x_axis)], 
                X_color_B[k:k_next,color_list.index(y_axis)], '.', alpha=0.5, label=set_names_B[i], color=plot_colors_B[i%len(plot_colors_B)])
        k = k_next

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    ax.legend()
    st.pyplot(fig)


with st.expander('Plot mit der Anzahl Lego-Steine eines Steine-Typs pro Achse'):
    x_axis = st.selectbox('X-Achse', part_types_names, index=part_types_names.index('Bricks'))
    y_axis = st.selectbox('Y-Achse', part_types_names, index=part_types_names.index('Plates'))

    fig, ax = plt.subplots()
    k = 0
    for i, X_ in enumerate(X_theme_list_A):
        k_next = k + X_.shape[0]
        ax.plot(X_theme_A[k:k_next,part_types_names.index(x_axis)], 
                X_theme_A[k:k_next,part_types_names.index(y_axis)], '.', alpha=0.5, label=set_names_A[i], color=plot_colors_A[i%len(plot_colors_A)])
        k = k_next
    k = 0
    for i, X_ in enumerate(X_theme_list_B):
        k_next = k + X_.shape[0]
        ax.plot(X_theme_B[k:k_next,part_types_names.index(x_axis)], 
                X_theme_B[k:k_next,part_types_names.index(y_axis)], '.', alpha=0.5, label=set_names_B[i], color=plot_colors_B[i%len(plot_colors_B)])
        k = k_next

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    ax.legend()
    st.pyplot(fig)

with st.expander('Set Namen und Nummern der gewählten Beispiele'):

    st.subheader('Gruppe A')
    st.table(pd.DataFrame({
                        'Name': set_names_A, 
                        'Nummer': set_nums_A}))

    st.subheader('Gruppe B')
    st.table(pd.DataFrame({
                        'Name': set_names_B, 
                        'Nummer': set_nums_B}))
