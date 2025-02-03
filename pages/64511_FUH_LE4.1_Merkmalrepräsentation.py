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
  

st.title('Merkmalrepräsentation')

# get data
photoset_path = os.path.join('data', 'photoset')

# raw_path = os.path.join(photoset_path, 'raw')
# if os.path.isdir(raw_path):
#     thumb_progress = st.progress(0.0, 'create thumbnails from raw ...')
#     raw_files = os.listdir(raw_path)
#     for i, file in enumerate(raw_files):
#         thumb_progress.progress(i/len(raw_files), 'create thumbnails from raw ...')
#         #thumb_path = os.path.join(photoset_path, f'{os.path.splitext(os.path.basename(file))[0]}.png')
#         thumb_path = os.path.join(photoset_path, f'{str(i+1).zfill(2)}.png')
#         if not os.path.isfile(thumb_path):
#             img = Image.open(os.path.join(photoset_path, 'raw', file))
#             thumb = img.resize((512, 512))
#             thumb.save(thumb_path)
#     thumb_progress.empty()

img_files_list = []
for file in os.listdir(photoset_path):
    img_path = os.path.join(photoset_path, file)
    if os.path.isfile(img_path):
        img_files_list.append(img_path.replace('\\', '/'))


# st.title('Merkmalextraktion und Repräsentation')
st.subheader('Extraktion')
st.write(r'''
    Jedes Datenobjekt (also in diesem Falle jedes Bild) wird Pixel für Pixel vom RGB Farbraum 
    in den HLS (**H**ue, **L**ightness, **S**aturation) Farbraum (Merkmalsraum/Feature Space $\mathbb{F} = \mathbb{R}^3$) transformiert.
''')

img_path = random.choice(img_files_list)
image = Image.open(img_path)
data = np.asarray(image)

R = data[:,:,0].flatten()
G = data[:,:,1].flatten()
B = data[:,:,2].flatten()

n = R.size
H = np.zeros(n)
L = np.zeros(n)
S = np.zeros(n)

fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1,2)
ax_img = fig.add_subplot(gs[0,0])
ax_F = fig.add_subplot(gs[0,1])

ax_img.imshow(image)
ax_img.set_title('Datenobjekt / Bild')

ax_F.set_title(r'Merkmalsraum $\mathbb{F}$')
ax_F.set_xlabel('Lightness / Helligkeit')
ax_F.set_ylabel('Saturation / Sättigung')

for i in range(n):
    H[i], L[i], S[i] = colorsys.rgb_to_hls(R[i]/255, G[i]/255, B[i]/255)
    if i%500 == 0:
        red, green, blue = colorsys.hls_to_rgb(H[i], 0.5, 1.0)
        # ax_F.scatter(L[i], S[i], color=(R[i]/255, G[i]/255, B[i]/255), edgecolor='#000000')
        ax_F.scatter(L[i], S[i], color=(red, green, blue), edgecolor='none')

st.pyplot(fig)

st.subheader('Gewichtung (Kernel)')

st.write(r'''
    Eine dominante Farbe darf nicht zu dunkel $L=0$ und nicht zu hell $L=1$ sein.
    Dies wird in der Gewichtung $w$ durch die Funktion $w_L(L, \alpha)$ berücksichtig.
    Um zusätzlich die Werte in der Mitte höher zu gewichten wird der Skalierungsfaktor $\alpha$ verwendet und der Wert von $w_L$ auf maximal $1$ begrenzt. 
''')
st.latex(r'''
    w_L(L, \alpha) = \left( \frac{1}{2} \sin(2 \pi L - \frac{\pi}{2}) + \frac{1}{2} \right) * \alpha
''')
def w_fun(L, y_cap_scale=1.5):
    return np.clip((np.sin(L * np.pi*2 - np.pi/2)/2 + 0.5) * y_cap_scale, 0.0, 1.0)

L_ = np.linspace(0.0, 1.0, 1000)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(L_, w_fun(L_, 1.0), 'k--', label=r'$\alpha=1.0$')
ax.plot(L_, w_fun(L_), 'k', label=r'$\alpha=1.5$')
ax.plot(L_, w_fun(L_, 2.0), 'k:', label=r'$\alpha=2.0$')
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$w_L(L, \alpha)$')
ax.legend()
st.pyplot(fig)
st.caption(r'''
    Beispiel der Funktion $w_L(L, \alpha)$ für $\alpha=1.0$, $\alpha=1.5$ und $\alpha=2.0$.
    Für das weitere generieren der Histogramme wurde $\alpha=1.5$ verwendet.
''')


st.write(r'''
    Die Sättigung $S$ geht im Quadrat in die Gewichtung $w(S, L) = S^2 \cdot w_L(L, \alpha=1.5)$ mit ein.
''')

st.subheader('Histogramm')

st.write(r'''
    Damit kann ein Histogramm über den Farbwert $H$ bestimmt werden, welches auch als Merkmalrepräsentation gespeichert wird. 
    Dabei wird die Sättigung $S$ und die Helligkeit $L$ als Gewichtung $w$ genutzt, um möglichst dominante Farben zu erkennen.
''')


fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2,1)
ax_F = fig.add_subplot(gs[0,0])
ax_hist = fig.add_subplot(gs[1,0])

w = np.zeros(n)
for i in range(n):
    w[i] = S[i]**2 * w_fun(L[i])
    if i%500 == 0:
        ax_F.scatter(H[i], w[i], color=(R[i]/255, G[i]/255, B[i]/255), edgecolor='#000000')

ax_F.set_xlim([0.0, 1.0])
ax_F.set_title('Merkmalsraum')
ax_F.set_ylabel(r'$w(S, L)$')

n_bins = 100
H_hist, H_bins = np.histogram(H, bins=n_bins, range=(0.0, 1.0), density=True, weights=w)

for k in range(n_bins):
    r, g, b = colorsys.hls_to_rgb(H_bins[k+1], 0.5, 1.0)
    ax_hist.bar(H_bins[k+1], H_hist[k], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b))

ax_hist.set_xlim([0.0, 1.0])
ax_hist.set_title('Histogramm')
ax_hist.set_xlabel(r'Farbwert $H$')
ax_hist.set_ylabel('Anzahl')

st.pyplot(fig)


feature_data_path = os.path.join(photoset_path, 'features')
if not os.path.exists(feature_data_path):
    os.makedirs(feature_data_path)
file_list_path = os.path.join(feature_data_path, 'file_list.csv')
histogram_data_path = os.path.join(feature_data_path, 'histogram_data.numpy')
save_hist_data = True


if os.path.isfile(histogram_data_path):
    st.stop()
else:
    histogram_data = np.zeros((len(img_files_list), n_bins))

# st.write(img_files_list)

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

        #w[i] = (np.sin(L[i]*np.pi*2 - np.pi/2)/2 + 0.5) * L[i] * S[i] * S[i] 
        w[i] = S[i]**2 * w_fun(L[i])

        #w[i] = w[i]**2

        if i%300 == 0:
            ax_px.scatter(H[i], w[i], color=(R[i]/255, G[i]/255, B[i]/255), edgecolor='#000000')

    ax_px.set_xlim([0.0, 1.0])
    ax_px.set_title('Merkmalsraum')
    ax_px.set_ylabel(r'$w(S, L)$')
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
    ax_hist.set_xlabel(r'Farbwert $H$')
    ax_hist.set_ylabel('Anzahl')

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