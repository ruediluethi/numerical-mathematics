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
from scipy.signal import find_peaks

st.title('Assoziationsregeln')


COLOR_WHEEL_NAMES = ['Rot', 'Orange', 'Gelb', 'Limette', 'Grün', 'Türkis', 'Cyan', 'Azurblau', 'Blau', 'Lila', 'Magenta', 'Purpur']
COLOR_VALUES = ['#FE2712', '#FC600A', '#FB9902', '#FCCC1A', '#FEFE33', '#B2D732', '#66B032', '#347C98', '#0247FE', '#4424D6', '#8601AF', '#C21460']


COLOR_WHEEL_NAMES = ['Rot', 'Gelb', 'Grün', 'Cyan', 'Blau', 'Magenta']


photoset_path = os.path.join('data', 'photoset')

n_bins = 100
feature_data_path = os.path.join(photoset_path, 'features')
histogram_data_path = os.path.join(feature_data_path, 'histogram_data.numpy')
if os.path.isfile(histogram_data_path):
    histogram_data = np.fromfile(histogram_data_path)
    n_photos = int(histogram_data.size/n_bins)
    histogram_data = np.reshape(histogram_data, (n_photos, n_bins))
    
if histogram_data is None:
    st.stop()

# get file list from saved csv
file_list_path = os.path.join(feature_data_path, 'file_list.csv')
df = pd.read_csv(file_list_path)
img_files_list = df['file'].to_numpy()

bar_width = 1.0/n_bins
hue = np.linspace(0.0, 1.0, n_bins+1)[0:-1]
n_cols = len(COLOR_WHEEL_NAMES)


data = []
show_all_hist_plots = True
for i, img_file in enumerate(img_files_list):

    if i%5 == 0:
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        rand_hist_i = random.randrange(5)
        hist_container = st.container()

    show_hist_plot = False
    if i%5 == rand_hist_i:
        if show_all_hist_plots:
            show_hist_plot = True


    crnt_column = cols[i%5]
    crnt_column.image(img_file)
    # continue
    # st.subheader(img_file)

    H_hist = histogram_data[i,:]
    H_hist = H_hist.reshape((n_bins,1))

    if show_hist_plot:
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1,3)
        ax_img = fig.add_subplot(gs[0,0])
        ax_hist = fig.add_subplot(gs[0, 1:3])

        image = Image.open(img_file)
        ax_img.imshow(image)

    peaks, props = find_peaks(H_hist.flatten(), distance=n_bins/n_cols/2, prominence=np.amax(H_hist)*0.01)
    

    col_borders = np.linspace(-1.0/n_cols/2, 1.0 + 1.0/n_cols/2, n_cols+2)
    col_names = []
    col_indices = []
    caption = ''
    for k, c in enumerate(col_borders):

        # if i%5 == rand_hist_i:
        #     if 0 < k and k <= n_cols:
        #         ax_hist.plot([c, c], [0, np.amax(H_hist)], 'k-')
        if k > 0:
            c_prev = col_borders[k-1]
            r, g, b = colorsys.hls_to_rgb((c_prev + c)/2, 0.5, 1.0)
            if show_hist_plot:
                ax_hist.fill([max(0.0, c_prev), max(0.0, c_prev), min(1.0, c), min(1.0, c)], 
                             [0.0, np.amax(H_hist), np.amax(H_hist), 0.0], color=(r, g, b), alpha=0.3)

            for p in peaks:
                if c_prev < hue[p+1] and hue[p+1] <= c:
                    col_names.append(COLOR_WHEEL_NAMES[(k-1)%n_cols])
                    col_indices.append((k-1)%n_cols)
                    
                    caption = f'{caption}<div style="float:left; width:15px; height:15px; margin: 5px 5px 0px 0px; background:rgb({r*255}, {g*255}, {b*255});"></div>'

    if show_hist_plot:
        for k in range(0, n_bins):
            r, g, b = colorsys.hls_to_rgb(hue[k], 0.5, 1.0)
            ax_hist.bar(hue[k]+bar_width, H_hist[k], width=bar_width, color=(r, g, b))

    if show_hist_plot:
        for peak in peaks:
            r, g, b = colorsys.hls_to_rgb(hue[peak+1], 0.5, 1.0)
            ax_hist.plot(hue[peak+1], H_hist[peak], 'o', markersize=12, color='#FFFFFF')
            ax_hist.plot(hue[peak+1], H_hist[peak], 'o', markersize=8, color=(r, g, b))

    if show_hist_plot:
        hist_container.pyplot(fig)
        if i <= 5:
            if not st.checkbox('Mehr Histogramme anzeigen'):
                show_all_hist_plots = False


    # st.write(col_names)
    data.append(col_indices)
    # crnt_column.caption(f'{", ".join(col_names)}')
    # crnt_column.write(col_indices)
    crnt_column.write(caption, unsafe_allow_html=True)


# data = [
#   ['A', 'B', 'E'],
#   ['B', 'D'],
#   ['B', 'C'],
#   ['A', 'B', 'D'],
#   ['A', 'B', 'C', 'E'],
#   ['A', 'B', 'C']
# ]

# st.write(data)

l1 = []
for d in data:
    for elem in d:
        if not elem in l1:
            l1.append(elem)

l1.sort()
lk = [[l] for l in l1]

def join(a, b):
    c = a.copy()
    for elem in b:
        if not elem in a:
            c.append(elem)
    c.sort()
    return c

def a_is_in_b(a, b):
    for elem in a:
        if not elem in b:
            return False
    return True
    
def count_a_in_b(a, b):
    fig, ax = plt.subplots()
    
    bar_counter = 0
    for k, i in enumerate(a):
        

        count = 0
        for j in b:
            if a_is_in_b(i, j):
                count += 1
        
        # st.write(i, count)
        
        # bar_width = 1.0
        if count > 1:
            # bar_width = 1.0/len(a)
            bar_counter += 1

            for kk, elem in enumerate(i):
                # col = COLOR_VALUES[elem]
                red, green, blue = colorsys.hls_to_rgb(elem/len(COLOR_WHEEL_NAMES), 0.5, 1.0)
                # ax.bar(k*bar_width + kk/len(i)*(bar_width*0.8), count, width=(bar_width*0.8)/len(i), color=(red, green, blue))
                ax.bar(bar_counter + 0.8/len(i)*kk, count, width=0.8/len(i), color=(red, green, blue))

    st.pyplot(fig)

    

count_a_in_b(lk, data)

for k in range(0, 3):
    
    lk_1 = []
    n = len(lk)
    for i in range(0, n):
        for j in range(i+1, n):
            do_join = True
            for l in range(0,len(lk[i])-1):
                if lk[i][l] != lk[j][l]:
                    do_join = False
                    break
            if do_join:
                lk_1.append(join(lk[i], lk[j]))
    
    lk = lk_1	
    
    count_a_in_b(lk, data)

