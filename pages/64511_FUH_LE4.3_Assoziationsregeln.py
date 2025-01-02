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


col1, col2, col3, col4, col5 = st.columns(5)
cols = [col1, col2, col3, col4, col5]
for i, img_file in enumerate(img_files_list):
    crnt_column = cols[i%5]
    crnt_column.image(img_file)
    # continue
    # st.subheader(img_file)

    H_hist = histogram_data[i,:]
    H_hist = H_hist.reshape((n_bins,1))

    # fig = plt.figure(figsize=(12, 4))
    # gs = fig.add_gridspec(1,3)
    # ax_img = fig.add_subplot(gs[0,0])
    # ax_hist = fig.add_subplot(gs[0, 1:3])

    # image = Image.open(img_file)
    # ax_img.imshow(image)

    for k in range(0, n_bins):
        r, g, b = colorsys.hls_to_rgb(hue[k], 0.5, 1.0)
        #ax_hist.bar(hue[k]+bar_width, H_hist[k], width=bar_width, color=(r, g, b))

    peaks, props = find_peaks(H_hist.flatten(), distance=n_bins/n_cols, prominence=np.amax(H_hist)*0.02)
    # for peak in peaks:
    #     r, g, b = colorsys.hls_to_rgb(hue[peak+1], 0.5, 1.0)
    #     ax_hist.plot(hue[peak+1], H_hist[peak], 'o', markersize=10, color=(r, g, b))

    col_borders = np.linspace(-1.0/n_cols/2, 1.0 + 1.0/n_cols/2, n_cols+2)
    col_names = []
    caption = ''
    for k, c in enumerate(col_borders):
        # if 0 < k and k <= n_cols:
            # ax_hist.plot([c, c], [0, np.amax(H_hist)], 'k-')
        if k > 0:
            for p in peaks:
                if col_borders[k-1] < hue[p+1] and hue[p+1] <= c:
                    col_names.append(COLOR_WHEEL_NAMES[(k-1)%n_cols])
                    r, g, b = colorsys.hls_to_rgb((col_borders[k-1] + c)/2, 0.5, 1.0)
                    caption = f'{caption}<div style="float:left; width:15px; height:15px; margin: 5px 5px 0px 0px; background:rgb({r*255}, {g*255}, {b*255});"></div>'

    # st.pyplot(fig)



    #crnt_column.caption(f'{", ".join(col_names)}')
    crnt_column.write(caption, unsafe_allow_html=True)