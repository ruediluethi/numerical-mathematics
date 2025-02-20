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

st.write('Datenquelle: https://www.kaggle.com/datasets/rtatman/lego-database')

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

# and group the colors into groups by there hue (and lightness/saturation)
def w_fun(L, y_cap_scale=1):
    return np.clip((np.sin(L * np.pi*2 - np.pi/2)/2 + 0.5) * y_cap_scale, 0.0, 1.0)
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
    return X

    X_ = np.zeros((X.shape[0], len(color_groups)))
    for i in range(len(color_groups)):
        for j in range(len(color_groups[i])):
            X_[:,i] += X[:,color_list.index(color_groups[i][j])]

    st.write(X_.shape)

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

plot_colors_A = ['red', 'purple', 'orange']
plot_colors_B = ['blue', 'green', 'cyan']
set_names_A = st.multiselect('Gruppe A besteht aus den Lego-Sets folgender Themen', themes_root.sort_values('count', ascending=False)['name'].to_list(), default=['Friends', 'Freestyle'])
set_names_B = st.multiselect('und Gruppe B aus diesen Themen', themes_root.sort_values('count', ascending=False)['name'].to_list(), default=['Ninjago', 'Bionicle'])

X_list_A = []
for set_name in set_names_A:
    root = themes_root[themes_root['name'] == set_name].iloc[0]
    X_list_A.append(get_X(root['root'], min_parts=50))
X_A = np.zeros((0, len(color_list)))
for X_ in X_list_A:
    X_A = np.vstack((X_A, X_))

X_list_B = []
for set_name in set_names_B:
    root = themes_root[themes_root['name'] == set_name].iloc[0]
    X_list_B.append(get_X(root['root'], min_parts=50))
X_B = np.zeros((0, len(color_list)))
for X_ in X_list_B:
    X_B = np.vstack((X_B, X_))

with st.expander('Plot mit der Anzahl Lego-Steine einer Farbe pro Achse'):
    x_axis = st.selectbox('X-Achse', color_list, index=color_list.index('Black'))
    y_axis = st.selectbox('Y-Achse', color_list, index=color_list.index('White'))

    fig, ax = plt.subplots()
    k = 0
    for i, X_ in enumerate(X_list_A):
        k_next = k + X_.shape[0]
        ax.plot(X_A[k:k_next,color_list.index(x_axis)], 
                X_A[k:k_next,color_list.index(y_axis)], '.', alpha=0.5, label=set_names_A[i], color=plot_colors_A[i])
        k = k_next
    k = 0
    for i, X_ in enumerate(X_list_B):
        k_next = k + X_.shape[0]
        ax.plot(X_B[k:k_next,color_list.index(x_axis)], 
                X_B[k:k_next,color_list.index(y_axis)], '.', alpha=0.5, label=set_names_B[i], color=plot_colors_B[i])
        k = k_next

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    ax.legend()
    st.pyplot(fig)


def PCA(A):
    n = A.shape[0]
    d = A.shape[1]
    ATA = 1/n * A.T @ A
    lambdas, V = np.linalg.eig(ATA)

    # st.write(lambdas)

    d = 3
    Sigma_ = np.zeros((n,d))
    Sigma_[:d,:d] = np.diag(np.sqrt(lambdas[:d]))

    # project the data onto the new 2D basis
    A_d = np.zeros((n,d))
    for i in range(0,d):
        A_d[:,i] = A @ V[:,i]

    return A_d

X = np.vstack((X_A, X_B))
A = PCA(X)
A_A = np.zeros((X_A.shape[0], A.shape[1]))
A_B = np.zeros((X_B.shape[0], A.shape[1]))

fig, ax = plt.subplots()
k = 0
for i, X_ in enumerate(X_list_A):
    k_next = k + X_.shape[0]
    ax.plot(A[k:k_next,0], A[k:k_next,1], 'o', alpha=0.5, label=set_names_A[i], color=plot_colors_A[i])
    A_A[k:k_next,:] = A[k:k_next,:]
    k = k_next
k_B = 0
for i, X_ in enumerate(X_list_B):
    k_next = k + X_.shape[0]
    k_B_next = k_B + X_.shape[0]
    ax.plot(A[k:k_next,0], A[k:k_next,1], 'o', alpha=0.5, label=set_names_B[i], color=plot_colors_B[i])
    A_B[k_B:k_B_next,:] = A[k:k_next,:]
    k = k_next
    k_B = k_B_next

# ax.plot(A_A[:,0], A_A[:,1], '.', label='Group A', color='red')
# ax.plot(A_B[:,0], A_B[:,1], '.', label='Group B', color='blue')

st.write('''
    Für jedes Legeo-Set wird die Anzahl der Steine pro Farbe gezählt und 
    durch die gesamt Anzahl der Steine eines Sets geteilt.
    So wird jedes Set durch einen Zeilenvektor $x_i$ repräsentiert,
    welcher als Merkmale die jeweilige prozentualen Anteile der Steinfarbe enthält.
''')

ax.legend()
st.pyplot(fig)
st.caption('''
    Die Lego-Sets aus den gewählten Themen werden mittels PCA auf die zwei (respektive drei) Hauptachsen projiziert
    und dem Thema entsprechend eingefärbt.
''')

st.write(r'''
    Die Matrix $X \in \mathbb{R}^{n \times D}$ enthält die Datenpunkte von $n$ Objekten (Zeilenvektor $x_i$) 
    mit $D$ Merkmalen/Dimensionen (Spalten - in diesem Falle die Hauptachsen).
''')

st.latex(r'''
    X = \left(\begin{array}{cccc}
        x_{11} & x_{12} & \cdots & x_{1D} \\
        \vdots & & \ddots & \vdots \\
        x_{n1} & x_{12} & \cdots & x_{nD} \\
    \end{array}\right)
''')

st.write(r'''
    Nun soll ein linearer Klassifikator $f$ mit den Gewichten $w \in \mathbb{R}^{D+1}$ bestimmt werden, 
    welcher die Daten der Matrix $X$ möglichst gut auf einen Zielvektor $y \in \mathbb{R}^n$ approximiert. 
    Damit der Klassifikator nicht nur skaliert, sondern auch eine Translation berücksichtigt, wird die Datenmatrix $X$ um eine Spalte erweitert
    und die Approximation kann als Matrixmultiplikation beschrieben werden:
''')

st.latex(r'''
    \left(\begin{array}{cccc}
        1 & x_{11} & x_{12} & \cdots & x_{1D} \\
        1 & \vdots & & \ddots & \vdots \\
        1 & x_{n1} & x_{12} & \cdots & x_{nD} \\
    \end{array}\right)
    \left(\begin{array}{c}
        w_0 \\
        \vdots \\
        w_D
    \end{array}\right) = \tilde{X}w
''')

st.write(r'''
    Nun sollen die Gewichte $w$ so gewählt werden, dass folgende Fehlerfunktion $J(w)$ minimal wird.      
''')

st.latex(r'''
    \left( y - \tilde{X}w \right)^\intercal  \left( y - \tilde{X}w \right)
    = \sum_{i=1}^{n} \left( y_i - \tilde{x}_i w \right)^2
    = J(w)
''')

st.write(r'''
    Die Fehlerfunktion $J(w)$ ist dann Minimal, wenn ihre Ableitung nach den Gewichten $w$ Null ist.
''')
st.latex(r'''
    \frac{\partial J(w)}{\partial w}
    = \frac{\partial}{\partial w} ( y^\intercal y \overbrace{- y^\intercal \tilde{X} w - w^\intercal \tilde{X}^\intercal y}^{
        \textrm{da } y^\intercal \tilde{X} w, w^\intercal \tilde{X}^\intercal y \in \mathbb{R} \quad \Rightarrow \quad 2w^\intercal \tilde{X}^\intercal y
    } + w^\intercal \tilde{X}^\intercal \tilde{X} w ) \\
    = 2 \tilde{X}^\intercal \tilde{X} w - 2 \tilde{X}^\intercal y
    \stackrel{!}{=} 0 \\
    \Leftrightarrow\quad w = \left( X^\intercal X \right)^{-1} X^\intercal y
''')

st.write(r'''
    Um nun einen neuen Datenpunkt $x_{\textrm{neu}}$ zu klassifizieren, muss dieser bloß mit den Gewichten $w$ multipliziert werden.
''')
st.latex(r'''
    f(x_{\textrm{neu}}) = \left( 1, x_{\textrm{neu},1}, ..., x_{\textrm{neu},D} \right) w = \hat{y}
''') 



def classify(X_a, X_b):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 3))
    X[0:X_a.shape[0],1:3] = X_a
    X[X_a.shape[0]:,1:3] = X_b

    y = np.ones(n)
    y[X_a.shape[0]:] = -1

    pseudo_inv = linalg.inv(X.T @ X) @ X.T
    w = pseudo_inv @ y

    fig, ax = plt.subplots()
    ax.plot(X[:,1][y==1], X[:,2][y==1], 'wo', markeredgecolor='k', label='Group A')
    ax.plot(X[:,1][y==-1], X[:,2][y==-1], 'k.', label='Group B')

    x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)
    border = (-w[0]-w[1]*x) / w[2]
    border_below = np.argwhere((X[:,2].min() < border) & (border < X[:,2].max()))
    ax.plot(x[border_below], border[border_below], 'k--', label='decision boundary')
    
    X_x_range = np.amax(X[:,1]) - np.amin(X[:,1])
    ax.set_xlim([np.amin(X[:,1]) - X_x_range*0.05, np.amax(X[:,1]) + X_x_range*0.05])
    X_y_range = np.amax(X[:,2]) - np.amin(X[:,2])
    ax.set_ylim([np.amin(X[:,2]) - X_y_range*0.05, np.amax(X[:,2]) + X_y_range*0.05])

    ax.legend()
    st.pyplot(fig)

classify(A_A[:,0:2], A_B[:,0:2])


st.write(r'''
    Die Trennlinie kann auch ein Polynom zweiten Grades sein.     
''')
st.latex(r'''
    X = \left(\begin{array}{ccccccc}
        x_{11} & x_{11}^2 & x_{11} & x_{11}^2 & \cdots & x_{1D} & x_{1D}^2 \\
        \vdots & & \ddots & \vdots \\
        x_{n1} & x_{n1}^2 & x_{12} & x_{12}^2 & \cdots & x_{nD} & x_{nD}^2 \\
    \end{array}\right)
''')


def classify_poly(X_a, X_b):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 3))
    X[0:X_a.shape[0],1:3] = X_a
    X[X_a.shape[0]:,1:3] = X_b

    X2 = np.ones((n, 5))
    X2[:,1] = X[:,1]
    X2[:,2] = X[:,1]**2
    X2[:,3] = X[:,2]
    X2[:,4] = X[:,2]**2

    # print(X2)

    y = np.ones(n)
    y[X_a.shape[0]:] = -1

    pseudo_inv = linalg.inv(X2.T @ X2) @ X2.T
    w = pseudo_inv @ y

    fig, ax = plt.subplots()
    ax.plot(X[:,1][y==1], X[:,2][y==1], 'wo', markeredgecolor='k', label='Group A')
    ax.plot(X[:,1][y==-1], X[:,2][y==-1], 'k.', label='Group B')

    x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)

    c = w[0] + w[1]*x + w[2]*x**2
    b = w[3]
    a = w[4]

    ax.plot(x, (-b + np.sqrt(b**2 - 4*a*c)) /(2*a), 'k--', label='decision boundary')
    ax.plot(x, (-b - np.sqrt(b**2 - 4*a*c)) /(2*a), 'k--')

    X_x_range = np.amax(X2[:,1]) - np.amin(X2[:,1])
    ax.set_xlim([np.amin(X2[:,1]) - X_x_range*0.05, np.amax(X2[:,1]) + X_x_range*0.05])
    X_y_range = np.amax(X2[:,3]) - np.amin(X2[:,3])
    ax.set_ylim([np.amin(X2[:,3]) - X_y_range*0.05, np.amax(X2[:,3]) + X_y_range*0.05])
    ax.legend()
    st.pyplot(fig)

classify_poly(A_A[:,0:2], A_B[:,0:2])



def classify_3d(X_a, X_b):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 4))
    X[:X_a.shape[0],1:4] = X_a
    X[X_a.shape[0]:,1:4] = X_b

    y = np.ones(n)
    y[X_a.shape[0]:] = -1


    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:X_a.shape[0],1], y=X[:X_a.shape[0],2], z=X[:X_a.shape[0],3],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Group A'
    ))
    fig.add_trace(go.Scatter3d(
        x=X[X_a.shape[0]:,1], y=X[X_a.shape[0]:,2], z=X[X_a.shape[0]:,3],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Group B'
    ))

    pseudo_inv = linalg.inv(X.T @ X) @ X.T
    w = pseudo_inv @ y
    # st.write(w)

    m = 10
    x = np.linspace(np.amin(X[:,1]), np.amax(X[:,1]), m)
    y = np.linspace(np.amin(X[:,2]), np.amax(X[:,2]), m)
    z = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            z[j,i] = (-w[0] - w[1]*x[i] - w[2]*y[j]) / w[3]
            if z[j,i] < np.amin(X[:,3]) or z[j,i] > np.amax(X[:,3]):
                z[j,i] = np.nan

    fig.add_trace(go.Surface(
        x=x, y=y, z=z
    ))
    st.plotly_chart(fig)


classify_3d(A_A[:,[0,2,1]], A_B[:,[0,2,1]])


# fig = go.Figure()


# k = 0
# for i, X_ in enumerate(X_list_A):
#     k_next = k + X_.shape[0]
#     fig.add_trace(go.Scatter3d(
#         x=A[k:k_next,0], y=A[k:k_next,1], z=A[k:k_next,2],
#         mode='markers',
#         marker=dict(size=5, color=plot_colors[i]),
#         name='Group A'
#     ))
#     k = k_next
# st.plotly_chart(fig)