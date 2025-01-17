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


st.title('Lineare Diskriminanzfunktion')

df_themes = pd.read_csv('data/lego/themes.csv')
df_themes = df_themes.set_index('id')

df_sets = pd.read_csv('data/lego/sets.csv')
df_sets_count = pd.DataFrame(df_sets.groupby('theme_id')['num_parts'].count().sort_values(ascending=False))
st.write(df_sets_count.join(df_themes))

df_inventories = pd.read_csv('data/lego/inventories.csv')
df_parts = pd.read_csv('data/lego/parts.csv')
# st.write(df_parts)
df_inv_parts = pd.read_csv('data/lego/inventory_parts.csv')

df_color_list = pd.read_csv('data/lego/colors.csv')
df_color_list = df_color_list.rename(columns={'id': 'color_id'})

st.write(df_color_list)
# st.write(df_color_list[df_color_list['name'].str.contains('Gray')])
gray_colors = df_color_list[(df_color_list['name'].str.contains('Black')) |
                            (df_color_list['name'].str.contains('Gray')) |
                            (df_color_list['name'].str.contains('Opaque'))
                        ]['name'].to_list()
st.write(gray_colors)

red_colors = df_color_list[
                            (df_color_list['name'].str.contains('Red')) | 
                            (df_color_list['name'].str.contains('Purple')) | 
                            (df_color_list['name'].str.contains('Pink')) |
                            (df_color_list['name'].str.contains('Lavender'))
                        ]['name'].to_list()

green_colors = df_color_list[
                            (df_color_list['name'].str.contains('Green')) | 
                            (df_color_list['name'].str.contains('Lime'))
                        ]['name'].to_list()
# purple_colors = df_color_list[(df_color_list['name'].str.contains('Red')) | (df_color_list['name'].str.contains('Green')) | (df_color_list['name'].str.contains('Blue'))]['name'].to_list()
# purple_colors = df_color_list[(df_color_list['name'].str.contains('Blue'))]['name'].to_list()
st.write(red_colors)

def get_X(theme_name):

    st.header(theme_name)

    all_set_nums = []
    all_set_names = []

    color_list = gray_colors + red_colors
    X = np.ones((0, len(color_list)))


    # st.write(df_themes[df_themes['name'] == theme_name])
    for theme_id, row in df_themes[df_themes['name'] == theme_name].iterrows():
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
    }))

    # st.write(np.array(color_list)[col_sort])
    # st.write(col_count[col_sort])

    # st.write(all_set_nums)
    # st.write(all_set_names)

    parts_count = X.sum(axis=1)[:, np.newaxis]
    X = X / parts_count

    st.write(col_names[0:len(gray_colors)])
    st.write(col_names[len(gray_colors):len(gray_colors) + len(red_colors)])

    st.write(X.shape)
    X_reduced = np.zeros((X.shape[0], 3))
    st.write(X_reduced.shape)
    X_reduced[:,0] = X[:,0:len(gray_colors)].sum(axis=1).flatten()
    X_reduced[:,1] = X[:,len(gray_colors):len(gray_colors) + len(red_colors)].sum(axis=1).flatten()
    X_reduced[:,2] = X[:,len(gray_colors) + len(red_colors):len(gray_colors) + len(red_colors) + len(green_colors)].sum(axis=1).flatten()

    count_filter = np.argwhere(parts_count > 70)[:,0]

    return X_reduced[count_filter,:] #, col_names, col_sort


X_friends = get_X('Friends')
# X_city = get_X('City')
X_starwars = get_X('Star Wars')

fig, ax = plt.subplots()
ax.plot(X_friends[:,0], X_friends[:,1], 'r.')
ax.plot(X_starwars[:,0], X_starwars[:,1], 'k.')
ax.set_xlabel('gray')
ax.set_ylabel('red')
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(X_friends[:,0], X_friends[:,2], 'r.')
ax.plot(X_starwars[:,0], X_starwars[:,2], 'k.')
ax.set_xlabel('gray')
ax.set_ylabel('green')
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(X_friends[:,1], X_friends[:,2], 'r.')
ax.plot(X_starwars[:,1], X_starwars[:,2], 'k.')
ax.set_xlabel('red')
ax.set_ylabel('green')
st.pyplot(fig)

st.stop()

#theme_name = 'Dinosaurs'
#theme_name = 'Train'

X, col, sort = get_X('Star Wars')

parts_count = X.sum(axis=1)[:, np.newaxis]
st.write(parts_count)

count_filter = np.argwhere(parts_count > 20)[:,0]

st.write(parts_count)
st.write(count_filter.shape)

X = X / parts_count

fig, ax = plt.subplots()
ax.plot(X[count_filter,0], X[count_filter,1], 'k.')
st.pyplot(fig)

for _, i in enumerate(sort):
    fig, ax = plt.subplots()
    ax.plot(X[count_filter,i], 'k.')
    ax.set_xlabel(col[i])
    st.pyplot(fig)



# fig, ax = plt.subplots()
# ax.plot(X[:,2], X[:,3], 'k.')
# st.pyplot(fig)


st.stop()


st.subheader('parts')
df_inv_parts = pd.read_csv('data/lego/inventory_parts.csv')
st.write(df_inv_parts)
st.write(df_inv_parts[df_inv_parts['inventory_id'] == 87])

df_color_list = pd.read_csv('data/lego/colors.csv')
df_color_list = df_color_list.set_index('id')
st.write(df_color_list)

df_colors = df_inv_parts.groupby('color_id')['quantity'].sum()
df_colors = pd.DataFrame(df_colors.sort_values(ascending=False))
st.write(df_colors.join(df_color_list))

df_inventories = df_inv_parts.groupby('inventory_id')['quantity'].sum()



st.write(df_inv_parts.groupby('inventory_id')['quantity'].sum())

st.stop()

# df_heros = pd.read_csv('data/superheroes_data.csv')



# df_male = df_heros[df_heros.gender == 'Male']
# df_female = df_heros[df_heros.gender == 'Female']



# st.write(df_female)

# st.write(df_male)

# fig, ax = plt.subplots()

# ax.plot(df_female['combat'], df_female['durability'], 'r.')

# ax.plot(df_male['combat'], df_male['durability'], 'b.')

# st.pyplot(fig)

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