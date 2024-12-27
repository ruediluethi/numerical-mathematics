import math
import os
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import colorsys

from scipy.signal import find_peaks

photoset_path = os.path.join('data', 'photoset')

for file in os.listdir(photoset_path):
#for file in ['fruits.png']:
    st.write(file)

    image = Image.open(os.path.join(photoset_path, file))

    st.image(image)

    data = np.asarray(image)

    R = data[:,:,0].flatten()
    G = data[:,:,1].flatten()
    B = data[:,:,2].flatten()

    n = R.size
    H = np.zeros(n)
    S = np.zeros(n)
    V = np.zeros(n)

    for i in range(n):
        H[i], S[i], V[i] = colorsys.rgb_to_hsv(R[i], G[i], B[i])

    V = V/255 # no idea why

    # H = (H + 1)%1.7

    n_bins = 100

    fig, ax = plt.subplots()
    H_hist, H_bins = np.histogram(H, bins=n_bins)
    for k in range(n_bins):
        r, g, b = colorsys.hsv_to_rgb(H_bins[k+1], 1.0, 1.0)
        ax.bar(H_bins[k+1], H_hist[k], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b))
    for k in range(int(n_bins*0.2)):
        r, g, b = colorsys.hsv_to_rgb(H_bins[k+1], 1.0, 1.0)
        ax.bar(H_bins[k+1]+1, H_hist[k], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b), alpha=0.5)


    n_cols = 12
    peaks, props = find_peaks(H_hist, prominence=np.amax(H_hist)*0.1, distance=n_bins/n_cols, width=int(n_bins*0.02))
    ax.plot(H_bins[peaks+1], H_hist[peaks], 'x')


    #col_names = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    col_names = ['red', 'orange', 'yellow', 'lime', 'green', 'blue-green', 'cyan', 'azure', 'blue', 'purple', 'magenta', 'pink']
    col_borders = np.linspace(-1.0/n_cols/2, 1.0 + 1.0/n_cols/2, n_cols+2)
    for k, c in enumerate(col_borders):
        ax.plot([c, c], [0, np.amax(H_hist)], 'k-')
        if k > 0:
            for p in peaks:
                if col_borders[k-1] < H_bins[p+1] and H_bins[p+1] <= c:
                    st.write(col_names[(k-1)%n_cols])

    st.pyplot(fig)

# for a in [H, S, V]:
#     hist, bins = np.histogram(a, bins=n_bin)

#     fig, ax = plt.subplots()

#     ax.bar(bins[1:], hist, width=np.abs(np.amax(bins) - np.amin(bins))/n_bin)

#     st.pyplot(fig)