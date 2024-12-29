import math
import os
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import colorsys

from scipy.signal import find_peaks
import random

COLOR_WHEEL_NAMES = ['Rot', 'Orange', 'Gelb', 'Limette', 'Grün', 'Türkis', 'Cyan', 'Azurblau', 'Blau', 'Lila', 'Magenta', 'Purpur']
    
st.title('Merkmalextraktion und Repräsentation')

photoset_path = os.path.join('data', 'photoset')

raw_path = os.path.join(photoset_path, 'raw')
if os.path.isdir(raw_path):
    thumb_progress = st.progress(0.0, 'create thumbnails from raw ...')
    raw_files = os.listdir(raw_path)
    for i, file in enumerate(raw_files):
        thumb_progress.progress(i/len(raw_files), 'create thumbnails from raw ...')
        #thumb_path = os.path.join(photoset_path, f'{os.path.splitext(os.path.basename(file))[0]}.png')
        thumb_path = os.path.join(photoset_path, f'{str(i+1).zfill(2)}.png')
        if not os.path.isfile(thumb_path):
            img = Image.open(os.path.join(photoset_path, 'raw', file))
            thumb = img.resize((512, 512))
            thumb.save(thumb_path)
    thumb_progress.empty()

img_files_list = []
for file in os.listdir(photoset_path):
    img_path = os.path.join(photoset_path, file)
    if os.path.isfile(img_path):
        img_files_list.append(img_path)

#st.write(len(img_files_list))

n_bins = 100
feature_data_path = os.path.join(photoset_path, 'features')
if not os.path.exists(feature_data_path):
    os.makedirs(feature_data_path)
histogram_data_path = os.path.join(feature_data_path, 'histogram_data.numpy')
save_hist_data = True
if os.path.isfile(histogram_data_path):
    histogram_data = np.fromfile(histogram_data_path).reshape((len(img_files_list), n_bins))
    img_files_list = [random.choice(img_files_list)]
    save_hist_data = False
    st.subheader('Beispiel der Merkmalextraktion eines Datenobjektes')
else:
    histogram_data = np.zeros((len(img_files_list), n_bins))

calc_hist_progress = st.progress(0.0, 'calc histogram data ...')
for i_file, img_path in enumerate(img_files_list):
    calc_hist_progress.progress(i_file/len(img_files_list), 'calc histogram data ...')


    image = Image.open(img_path)

    data = np.asarray(image)

    R = data[:,:,0].flatten()
    G = data[:,:,1].flatten()
    B = data[:,:,2].flatten()

    n = R.size
    H = np.zeros(n)
    S = np.zeros(n)
    V = np.zeros(n)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2,3)
    ax_img = fig.add_subplot(gs[:,0])
    ax_px = fig.add_subplot(gs[0, 1:3])
    ax_hist = fig.add_subplot(gs[1, 1:3])

    ax_img.imshow(image)
    ax_img.set_title('Datenobjekt')    

    for i in range(n):
        H[i], S[i], V[i] = colorsys.rgb_to_hsv(R[i], G[i], B[i])
        w = S[i]*(V[i]/255)
        if i%200 == 0:
            ax_px.plot(H[i], w, '.', color=(R[i]/255, G[i]/255, B[i]/255))
    ax_px.set_xlim([0.0, 1.0])
    ax_px.set_title('Merkmalsraum')
    # st.pyplot(fig)

    V = V/255 # no idea why


    # fig, ax = plt.subplots()
    H_hist, H_bins = np.histogram(H, bins=n_bins, density=True, weights=S*V)
    if save_hist_data:
        histogram_data[i_file,:] = H_hist
    for k in range(n_bins):
        r, g, b = colorsys.hsv_to_rgb(H_bins[k+1], 1.0, 1.0)
        ax_hist.bar(H_bins[k+1], H_hist[k], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b))
    # for k in range(int(n_bins*0.2)):
    #     r, g, b = colorsys.hsv_to_rgb(H_bins[k+1], 1.0, 1.0)
    #     ax.bar(H_bins[k+1]+1, H_hist[k], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b), alpha=0.5)


    n_cols = len(COLOR_WHEEL_NAMES)
    peaks, props = find_peaks(H_hist, prominence=np.amax(H_hist)*0.1, distance=n_bins/n_cols/2, width=int(n_bins*0.01))
    ax_hist.plot(H_bins[peaks+1], H_hist[peaks], 'kx', markersize=10)


    col_borders = np.linspace(-1.0/n_cols/2, 1.0 + 1.0/n_cols/2, n_cols+2)
    col_names = []
    for k, c in enumerate(col_borders):
        if 0 < k and k <= n_cols:
            ax_hist.plot([c, c], [0, np.amax(H_hist)], 'k-')
        if k > 0:
            for p in peaks:
                if col_borders[k-1] < H_bins[p+1] and H_bins[p+1] <= c:
                    col_names.append(COLOR_WHEEL_NAMES[(k-1)%n_cols])

    ax_hist.set_xlim([0.0, 1.0])
    ax_hist.set_title('Histogramm')

    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f'Merkmalsignatur: {", ".join(col_names)}')

calc_hist_progress.empty()
if save_hist_data:
    histogram_data.tofile(histogram_data_path)

# st.write(histogram_data.shape)
# st.write(histogram_data)


# for a in [H, S, V]:
#     hist, bins = np.histogram(a, bins=n_bin)

#     fig, ax = plt.subplots()

#     ax.bar(bins[1:], hist, width=np.abs(np.amax(bins) - np.amin(bins))/n_bin)

#     st.pyplot(fig)