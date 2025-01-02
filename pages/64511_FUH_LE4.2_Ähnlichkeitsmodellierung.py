import os
import streamlit as st
import numpy as np
import numpy.linalg as linalg

from PIL import Image, ImageColor

import matplotlib.pyplot as plt
import random

from scipy.stats import norm
import colorsys
import pandas as pd
import math

st.title('Ähnlichkeitsmodellierung')
st.write('Für jede der 12 Farben des RYB Farbkreises soll das Bild mit dem ähnlichsten Histogramm gefunden werden. Dafür werden zwei verschiedene Ähnlichkeitsfunktionen verglichen.')
st.image('static/pic_ryb_itten.jpg')
st.caption('Der RYB Farbkreises. Quelle: https://www.w3schools.com/colors/colors_wheels.asp')

#COLOR_WHEEL_NAMES = ['Rot', 'Orange', 'Gelb', 'Limette', 'Grün', 'Türkis', 'Cyan', 'Azurblau', 'Blau', 'Lila', 'Magenta', 'Purpur']

COLOR_WHEEL_NAMES = ['RED', 'RED-ORANGE', 'ORANGE', 'YELLOW-ORANGE', 'YELLOW', 'YELLOW-GREEN', 'GREEN', 'BLUE-GREEN', 'BLUE', 'BLUE-PURPLE', 'PURPLE', 'RED-PURPLE']
COLOR_VALUES = ['#FE2712', '#FC600A', '#FB9902', '#FCCC1A', '#FEFE33', '#B2D732', '#66B032', '#347C98', '#0247FE', '#4424D6', '#8601AF', '#C21460']


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



n_cols = len(COLOR_WHEEL_NAMES)
s_width = 1/n_cols/2
for j in range(0, n_cols):
#for j in [0,5]:

    st.subheader(COLOR_WHEEL_NAMES[j])

    col_r, col_g, col_b = ImageColor.getcolor(COLOR_VALUES[j], "RGB")
    # st.write(r, g, b)
    col_h, col_l, col_s = colorsys.rgb_to_hls(col_r/255, col_g/255, col_b/255)
    # st.write(h, l, s)
    
    
    # mu = 1.0/n_cols * j
    mu = col_h
    sigma = s_width/3

    x = np.linspace(-1.0, 2.0, n_bins*3)
    col_hist_pdf = norm.pdf(x, mu, sigma)

    for i in range(0, n_bins):
        if col_hist_pdf[i] > 0.001:
            col_hist_pdf[i+n_bins] = col_hist_pdf[i]
    for i in range(n_bins*2, n_bins*3):
        if col_hist_pdf[i] > 0.001:
            col_hist_pdf[i-n_bins] = col_hist_pdf[i]
    x = x[n_bins:n_bins*2]
    col_hist_pdf = col_hist_pdf[n_bins:n_bins*2].reshape((n_bins, 1))
    #col_hist = col_hist / np.amax(col_hist)

    col_hist_peak = np.zeros(n_bins)
    # col_hist_peak[int(j/n_cols*n_bins)] = 100.0
    mu_i = int(math.floor(mu*n_bins))
    col_hist_peak[mu_i] = 1.0

    bar_width = 1.0/n_bins/2

    hue = np.linspace(0.0, 1.0, n_bins+1)[0:-1]

    similarity_norm = np.zeros(n_photos)
    similarity_sqfd = np.zeros(n_photos)


    for i in range(n_photos):
    
        photo_hist = histogram_data[i,:]
        photo_hist = photo_hist.reshape((n_bins,1))

        # dot product
        # similarity[i] = photo_hist.T @ col_hist / (linalg.norm(photo_hist)*linalg.norm(col_hist))
        
        # euclidean distance
        similarity_norm[i] = linalg.norm(photo_hist - col_hist_pdf)

        # signature quadratic form distanz
        similarity_sqfd[i] = 0.0
        

        # for f in range(0, n_bins): # loop not necessary because only one peak in col_hist_peak is not zero 
        f = mu_i
        for g in range(0, n_bins):
            
            s = 1 - min(s_width, 
                        abs(hue[f] - hue[g]), 
                        abs(hue[f] - (hue[g]-1)),
                        abs(hue[f] - (hue[g]+1)) 
                    ) / s_width
            # s = s**2
            
            similarity_sqfd[i] = similarity_sqfd[i] + col_hist_peak[f] * photo_hist[g] * s

    # st.pyplot(fig)

    sim_sorted_norm = np.argsort(similarity_norm)
    sim_sorted_sqfd = np.argsort(similarity_sqfd)[::-1]

    tab1, tab2 = st.tabs(['Signature Quadratic Form Distanz', 'Euklidische Distanz'])
    tabs = [tab2, tab1]

    col_hist = [col_hist_pdf, col_hist_peak]
    for t, sim_sorted in enumerate([sim_sorted_norm, sim_sorted_sqfd]):
        for i in range(0, 3):

            fig = plt.figure(figsize=(14, 2))
            gs = fig.add_gridspec(1,3)
            ax_img = fig.add_subplot(gs[0,0])
            ax_px = fig.add_subplot(gs[0, 1:3])

            image = Image.open(img_files_list[sim_sorted[i]])
            ax_img.imshow(image)
            ax_img.set_title(sim_sorted[i])

            photo_hist = histogram_data[sim_sorted[i],:].reshape((n_bins,1))
            #photo_hist = photo_hist / np.amax(photo_hist)

            
            if t == 1:
                ax_px.bar(hue[f], np.amax(photo_hist), width=bar_width, color=COLOR_VALUES[j])

            for g in range(0, n_bins):
                
                s = 1 - min(s_width, 
                        abs(hue[f] - hue[g]), 
                        abs(hue[f] - (hue[g]-1)),
                        abs(hue[f] - (hue[g]+1)) 
                    ) / s_width
                # s = s**2

                red, green, blue = colorsys.hls_to_rgb(hue[g], 0.5, 1.0)
                if t == 0:
                    ax_px.bar(hue[g], col_hist[t][g], width=bar_width, color=COLOR_VALUES[j])#, color=(r, g, b))

                
                
                if t == 1:
                    ax_px.bar(hue[g]+bar_width, photo_hist[g], width=bar_width, color=(red, green, blue, 0.2))
                    ax_px.bar(hue[g]+bar_width, photo_hist[g]*s, width=bar_width, color=(red, green, blue))
                else:
                    ax_px.bar(hue[g]+bar_width, photo_hist[g], width=bar_width, color=(red, green, blue))

            # ax.plot(x, y)

            tabs[t].pyplot(fig)