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
df_themes = df_themes.set_index('id')

df_sets = pd.read_csv('data/lego/sets.csv')
df_sets_count = pd.DataFrame(df_sets.groupby('theme_id')['num_parts'].count().sort_values(ascending=False))
# st.write(df_sets_count.join(df_themes))

df_inventories = pd.read_csv('data/lego/inventories.csv')
df_parts = pd.read_csv('data/lego/parts.csv')
df_inv_parts = pd.read_csv('data/lego/inventory_parts.csv')

df_color_list = pd.read_csv('data/lego/colors.csv')
df_color_list = df_color_list.rename(columns={'id': 'color_id'})
st.write(df_color_list)

def w_fun(L, y_cap_scale=1):
    return np.clip((np.sin(L * np.pi*2 - np.pi/2)/2 + 0.5) * y_cap_scale, 0.0, 1.0)

fig, ax = plt.subplots()
for i, row in df_color_list.iterrows():
    r, g, b = mcolors.hex2color('#'+row['rgb'])
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    #ax.plot([h], [s**2*w_fun(l)], 'o', color=(r, g, b))

    ax.plot([h], [l], '.', color=(r, g, b), markersize=10)

    if l < 0.15:
        ax.plot([h], [l], 'wx')
        st.write(row['name'])
    elif l > 0.9:
        #ax.plot([h], [l], 'X', color=(r, g, b), markersize=10)
        ax.plot([h], [l], 'kx')
    elif s < 0.15:
        ax.plot([h], [l], 'k+')

    elif h < 0.05:
        ax.plot([h], [l], '.', color=(r, g, b), markersize=10)
    elif h < 0.1:
        ax.plot([h], [l], '.', color=(r, g, b), markersize=20)
    

st.pyplot(fig)

st.stop()

gray_colors = df_color_list[(df_color_list['name'].str.contains('Black')) |
                            (df_color_list['name'].str.contains('Gray')) |
                            (df_color_list['name'].str.contains('Opaque'))
                        ]['name'].to_list()
red_colors = df_color_list[
                            (df_color_list['name'].str.contains('Red')) | 
                            (df_color_list['name'].str.contains('Purple')) | 
                            (df_color_list['name'].str.contains('Pink')) |
                            (df_color_list['name'].str.contains('Lavender')) |
                            (df_color_list['name'].str.contains('Magenta')) |
                            (df_color_list['name'].str.contains('Orange')) |
                            (df_color_list['name'].str.contains('Yellow'))
                        ]['name'].to_list()

# red_colors = df_color_list[
#                             (df_color_list['name'].str.contains('Blue')) | 
#                             (df_color_list['name'].str.contains('Azure'))
#                         ]['name'].to_list()

green_colors = df_color_list[
                            (df_color_list['name'].str.contains('Green')) | 
                            (df_color_list['name'].str.contains('Lime'))
                        ]['name'].to_list()


def get_X(theme_name, min_parts=100):

    st.header(theme_name)

    all_set_nums = []
    all_set_names = []

    color_list = gray_colors + red_colors
    X = np.ones((0, len(color_list)))


    # st.write(df_themes[df_themes['name'].str.contains(theme_name, case=False)])
    for theme_id, row in df_themes[df_themes['name'].str.contains(theme_name, case=False)].iterrows():
        # st.subheader(theme_id)
        df_theme_sets = df_sets[df_sets['theme_id'] == theme_id]
        df_theme_sets = pd.merge(df_theme_sets, df_inventories, on='set_num')
        # st.write(df_theme_sets)
        for i, row in df_theme_sets.iterrows():
            
            df_parts_set = df_inv_parts[df_inv_parts['inventory_id'] == row['id']]
            df_colors = pd.DataFrame(df_parts_set.groupby('color_id')['quantity'].sum()).reset_index()
            df_colors = pd.merge(df_colors, df_color_list, on='color_id')
            if df_colors.shape[0] > 0:
                # st.write(f'{row['set_num']}: {row['name']} (id: {row['id']})')
                # st.write(df_colors)

                X = np.vstack((X, np.zeros((1, len(color_list)))))
                all_set_nums.append(row['set_num'])
                all_set_names.append(row['name'])

                for i, row_col in df_colors.iterrows():
                    if row_col['name'] not in color_list:
                        color_list.append(row_col['name'])
                        X = np.hstack((X, np.zeros((len(all_set_nums), 1))))

                    X[len(all_set_nums)-1, color_list.index(row_col['name'])] = row_col['quantity']

                


    # st.write(X)
    # st.write(X.shape)
    # st.write(color_list)
    col_names = np.array(color_list)
    col_count = np.sum(X, axis=0)
    col_sort = np.argsort(col_count)[::-1]

    

    st.write(pd.DataFrame({
        'color': col_names[col_sort],
        'count': col_count[col_sort]
    }).head(20))

    # st.write(np.array(color_list)[col_sort])
    # st.write(col_count[col_sort])

    # st.write(all_set_nums)
    # st.write(all_set_names)

    parts_count = X.sum(axis=1)[:, np.newaxis]
    X = X / parts_count

    # st.write(col_names[0:len(gray_colors)])
    # st.write(col_names[len(gray_colors):len(gray_colors) + len(red_colors)])

    X_reduced = np.zeros((X.shape[0], 3))
    X_reduced[:,0] = X[:,0:len(gray_colors)].sum(axis=1).flatten()
    X_reduced[:,1] = X[:,len(gray_colors):len(gray_colors) + len(red_colors)].sum(axis=1).flatten()
    X_reduced[:,2] = X[:,len(gray_colors) + len(red_colors):len(gray_colors) + len(red_colors) + len(green_colors)].sum(axis=1).flatten()

    count_filter = np.argwhere(parts_count > min_parts)[:,0]

    X_reduced = X_reduced[count_filter,:]
    # X_reduced = X_reduced / X_reduced.sum(axis=1)[:, np.newaxis]

    return X_reduced

    col_min = X_reduced.min(axis=0)
    col_max = X_reduced.max(axis=0)

    return (X_reduced - col_min) / (col_max - col_min)


X_friends = get_X('Friends', min_parts=50)
X_starwars = get_X('Star Wars', min_parts=200)

st.write(X_friends.shape)
st.write(X_starwars.shape)

def classify(X_a, X_b, label_a=None, label_b=None):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 3))
    X[0:X_a.shape[0],1:3] = X_a
    X[X_a.shape[0]:,1:3] = X_b

    y = np.ones(n)
    y[X_a.shape[0]:] = -1

    pseudo_inv = linalg.inv(X.T @ X) @ X.T
    w = pseudo_inv @ y

    fig, ax = plt.subplots()
    ax.plot(X[:,1][y==1], X[:,2][y==1], 'wo', markeredgecolor='k', label='Friends')
    ax.plot(X[:,1][y==-1], X[:,2][y==-1], 'k.')

    x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)
    border = (-w[0]-w[1]*x) / w[2]
    border_below = np.argwhere((X[:,2].min() < border) & (border < X[:,2].max()))
    ax.plot(x[border_below], border[border_below], 'k--', label='decision boundary')
    
    if label_a is not None:
        ax.set_xlabel(label_a)
    if label_b is not None:
        ax.set_ylabel(label_b)

    ax.set_aspect('equal')

    st.pyplot(fig)

def classify_poly(X_a, X_b, label_a=None, label_b=None):
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
    ax.plot(X[:,1][y==1], X[:,2][y==1], 'wo', markeredgecolor='k', label='ok')
    ax.plot(X[:,1][y==-1], X[:,2][y==-1], 'k.', label='nok')

    x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)

    c = w[0] + w[1]*x + w[2]*x**2
    b = w[3]
    a = w[4]

    ax.plot(x, (-b + np.sqrt(b**2 - 4*a*c)) /(2*a), 'r--', label='decision boundary')
    ax.plot(x, (-b - np.sqrt(b**2 - 4*a*c)) /(2*a), 'b--', label='decision boundary')

    # border = (-w[0]-w[1]*x) / w[2]
    # border_below = np.argwhere((X[:,2].min() < border) & (border < X[:,2].max()))
    # ax.plot(x[border_below], border[border_below], 'k--', label='decision boundary')
    
    if label_a is not None:
        ax.set_xlabel(label_a)
    if label_b is not None:
        ax.set_ylabel(label_b)

    res = 300
    J = np.zeros((res, res))
    for i, x1 in enumerate(np.linspace(X[:,1].min(), X[:,1].max(), res)):
        for j, x2 in enumerate(np.linspace(X[:,2].min(), X[:,2].max(), res)):
            J[j,i] = w[0] + w[1]*x1 + w[2]*x1**2 + w[3]*x2 + w[4]*x2**2
    
    J[J < 0] = J[J < 0] / np.min(J)
    J[J > 0] = J[J > 0] / np.max(J)
    J = (1 - J)**10

    # ax.set_aspect('equal')

    st.pyplot(fig)

classify_poly(X_friends[:,[1,0]], X_starwars[:,[1,0]], 'red', 'gray')
classify_poly(X_friends[:,[0,2]], X_starwars[:,[0,2]], 'gray', 'green')
classify_poly(X_friends[:,1:3], X_starwars[:,1:3], 'red', 'green')

def classify_3d(X_a, X_b, label_a=None, label_b=None, label_c=None):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 4))
    X[0:X_a.shape[0],1:4] = X_a
    X[X_a.shape[0]:,1:4] = X_b

    y = np.ones(n)
    y[X_a.shape[0]:] = -1

    st.write(X)
    st.write(X.T @ X)

    

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X_a[:,0], y=X_a[:,1], z=X_a[:,2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Group A'
    ))
    fig.add_trace(go.Scatter3d(
        x=X_b[:,0], y=X_b[:,1], z=X_b[:,2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Group B'
    ))

    pseudo_inv = linalg.inv(X.T @ X) @ X.T
    w = pseudo_inv @ y
    st.write(w)

    m = 100
    x_1 = np.linspace(X[:,1].min(), X[:,1].max(), m)
    x_2 = np.linspace(X[:,2].min(), X[:,2].max(), m)
    x_3 = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            x_3[i,j] = (-w[0] - w[1]*x_1[i] - w[2]*x_2[j]) / w[3]
            if x_3[i,j] < X[:,3].min() or x_3[i,j] > X[:,3].max():
                x_3[i,j] = np.nan

    fig.add_trace(go.Surface(
        x=x_1, y=x_2, z=x_3
    ))
    st.plotly_chart(fig)

    # st.plotly_chart(px.scatter_3d(pd.DataFrame({
    #     'x': X[:,1],
    #     'y': X[:,2],
    #     'z': X[:,3],
    #     'c': y
    # }), x='x', y='y', z='z', color='c'))


classify_3d(X_friends, X_starwars)

n = 100
X_test_a = np.zeros((n,3)) + np.random.rand(n,3) * 2
n = 50
X_test_b = np.ones((n,3)) + np.random.rand(n,3) * 1

classify_3d(X_test_a, X_test_b)

st.header('dummy example')


df = pd.DataFrame({ 
    'age': [23, 17, 43, 68, 32],
    'max_speed': [180, 240, 246, 173, 110],
    'risk': [1, 1, 1, -1, -1]
})

st.write(df)

age = df['age'].to_numpy()
max_speed = df['max_speed'].to_numpy()
y = df['risk'].to_numpy()

n = y.size

st.subheader('Zweidimensional')

X = np.ones((n,3))
X[:,1] = age
X[:,2] = max_speed

st.write(X)

pseudo_inv = linalg.inv(X.T @ X) @ X.T

st.write(pseudo_inv)

w = pseudo_inv @ y



st.write(w)

st.write(y - X @ w)

fig, ax = plt.subplots()
ax.plot(age[y==1], max_speed[y==1], 'ro', label='high risk')
ax.plot(age[y==-1], max_speed[y==-1], 'go', label='low rist')

ax.plot(age, (-w[0]-w[1]*age)/w[2], 'b-', label='decision boundary')

st.pyplot(fig)


st.subheader('Eindimensional')

n = 100

n_high = int(n*0.25 + random.randint(0, int(n*0.5)))
n_low = n - n_high

st.write(n, n_high, n_low, n_high + n_low)

high = np.random.normal(loc=3, scale=0.5, size=n_high)
low = np.random.normal(loc=1, scale=0.5, size=n_low)

y = np.concatenate((np.ones(n_high), -np.ones(n_low)))
X = np.ones((n,2))
X[:,1] = np.concatenate((high, low))

pseudo_inv = linalg.inv(X.T @ X) @ X.T
w = pseudo_inv @ y

fig, ax = plt.subplots()
ax.plot(np.zeros(n_high), X[:,1][y==1], 'ro', label='high risk')
ax.plot(np.zeros(n_low), X[:,1][y==-1], 'g.', label='low risk')

ax.plot([-1, 1], np.ones(2) * -w[0]/w[1], 'b-', label='decision boundary')


st.pyplot(fig)