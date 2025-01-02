import os
import streamlit as st
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import colorsys

from scipy.signal import find_peaks

import random
import pandas as pd

COLOR_WHEEL_NAMES = ['Rot', 'Orange', 'Gelb', 'Limette', 'Grün', 'Türkis', 'Cyan', 'Azurblau', 'Blau', 'Lila', 'Magenta', 'Purpur']
  

# x = np.linspace(0.0, 1.0, 100)

# fig, ax = plt.subplots()

# ax.plot(x, (np.sin(x*np.pi*2 - np.pi/2)/2 + 0.5) * x)

# st.pyplot(fig)

# st.stop()

st.title('Merkmalextraktion und Repräsentation')
st.write('Für die Merkmalsextraktion wurde Pixel für Pixel vom RGB Farbraum in den HSL (Hue, Saturation, Lightness) Farbraum transformiert. Damit kann ein Histogramm über den Farbwert (H) bestimmt werden, welches auch als Merkmalrepräsentation gespeichert wird. Dabei wird die Sättigung (S) und die Helligkeit (L) als Gewichtung w genutzt, um möglichst dominante Farben zu erkennen. Denn eine Farbe darf nicht zu dunkel (L=0) und nicht zu hell (L=1) sein, sowie eine möglichst hohe Sättigung S aufweisen. Deshalb wird die folgende Funktion zur Bestimmung der Gewichtung w verwendet:')
st.latex(r'''
    w(S, L) = \left( \frac{1}{2} \sin(2 \pi L - \frac{\pi}{2}) + \frac{1}{2} \right) \cdot L \cdot S^2
''')

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
file_list_path = os.path.join(feature_data_path, 'file_list.csv')
histogram_data_path = os.path.join(feature_data_path, 'histogram_data.numpy')
save_hist_data = True
if os.path.isfile(histogram_data_path):
    histogram_data = np.fromfile(histogram_data_path).reshape((len(img_files_list), n_bins))
    img_files_list = [random.choice(img_files_list)]
    # img_files_list = [img_files_list[25]]
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
    L = np.zeros(n)
    w = np.zeros(n)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2,3)
    ax_img = fig.add_subplot(gs[:,0])
    ax_px = fig.add_subplot(gs[0, 1:3])
    ax_hist = fig.add_subplot(gs[1, 1:3])

    ax_img.imshow(image)
    ax_img.set_title('Datenobjekt')    

    for i in range(n):
        #H[i], S[i], V[i] = colorsys.rgb_to_hsv(R[i], G[i], B[i])
        #w = S[i]*(V[i]/255)
        H[i], L[i], S[i] = colorsys.rgb_to_hls(R[i]/255, G[i]/255, B[i]/255)
        #w[i] = 1-(L[i]-0.5)**2 * S[i]

        w[i] = (np.sin(L[i]*np.pi*2 - np.pi/2)/2 + 0.5) * L[i] * S[i] * S[i] 

        #w[i] = w[i]**2

        if i%300 == 0:
            ax_px.scatter(H[i], w[i], color=(R[i]/255, G[i]/255, B[i]/255), edgecolor='#000000')

    ax_px.set_xlim([0.0, 1.0])
    ax_px.set_title('Merkmalsraum')
    # st.pyplot(fig)


    # fig, ax = plt.subplots()
    H_hist, H_bins = np.histogram(H, bins=n_bins, range=(0.0, 1.0), density=True, weights=w)
    # st.write(H_bins)
    # st.write(H_hist)
    if save_hist_data:
        histogram_data[i_file,:] = H_hist
    for k in range(n_bins):
        r, g, b = colorsys.hls_to_rgb(H_bins[k+1], 0.5, 1.0)
        ax_hist.bar(H_bins[k+1], H_hist[k], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b))
    
    ax_hist.set_xlim([0.0, 1.0])
    ax_hist.set_title('Histogramm')

    fig.tight_layout()
    st.pyplot(fig)
    st.caption('Beispielbild mit transformierten Pixeln und dazugehörigen gewichteten Histogramm.')

calc_hist_progress.empty()
if save_hist_data:
    histogram_data.tofile(histogram_data_path)

    df = pd.DataFrame(data={"file": img_files_list})
    df.to_csv(file_list_path, sep=',',index=True)

# st.write(histogram_data.shape)
# st.write(histogram_data)


# for a in [H, S, V]:
#     hist, bins = np.histogram(a, bins=n_bin)

#     fig, ax = plt.subplots()

#     ax.bar(bins[1:], hist, width=np.abs(np.amax(bins) - np.amin(bins))/n_bin)

#     st.pyplot(fig)