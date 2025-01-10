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
st.write('Für jede der 12 Farben des RYB Farbkreises soll das Bild mit dem ähnlichsten Histogramm gefunden werden.')
col1, col2, col3 = st.columns(3)
col2.image('static/pic_ryb_itten.jpg')
st.caption('Der RYB Farbkreises. Quelle: https://www.w3schools.com/colors/colors_wheels.asp')
COLOR_WHEEL_NAMES = ['RED', 'RED-ORANGE', 'ORANGE', 'YELLOW-ORANGE', 'YELLOW', 'YELLOW-GREEN', 'GREEN', 'BLUE-GREEN', 'BLUE', 'BLUE-PURPLE', 'PURPLE', 'RED-PURPLE']
COLOR_VALUES = ['#FE2712', '#FC600A', '#FB9902', '#FCCC1A', '#FEFE33', '#B2D732', '#66B032', '#347C98', '#0247FE', '#4424D6', '#8601AF', '#C21460']


st.write(r'''
    Dafür werden drei verschiedene Ähnlichkeitsfunktionen zwischen 
    den Histogrammen $X, Y \in \mathbb{H}_R$ mit den Bins (oder Unterteilungen) $R=\{ r_i \}_{i=1}^n$ verwendet.
    Dabei bezeichnet $X(r_i)$ den Wert (meist die Anzahl oder Amplitude) des Histogrammes $X$ bei Bin $r$ und
    die dazugehörigen Gewichtsvektoren $x, y$ zu den Histogrammen $X, Y$ sind wie folgt definiert:
''')
st.latex(r'''
    x = \left( X(r_1) \cdots X(r_n) \right) \\
    y = \left( Y(r_1) \cdots Y(r_n) \right)
''')
st.write(r'''
    Die Distanz zwischen zwei Bins wird durch die Ähnlichkeitsfunktion $s: R \times R \rightarrow \mathbb{R}$ bestimmt.
    Für die weiteren Beispiele wird eine lineare Ähnlichkeitsfunktion $s$ zwischen den Bins $f, g \in R$ mit einer maximalen Distanz $s_{max}$ verwendet:
''')
st.latex(r'''
    s(f, g) = 1 - \frac{|f - g|}{s_{max}}
''')
n_cols = len(COLOR_WHEEL_NAMES)
s_width = 1/n_cols
def s_fun(f, g, s_width=1/n_cols/2):
    return 1 - min(s_width, 
            abs(f - g), 
            abs(f - (g-1)),
            abs(f - (g+1)) 
        ) / s_width

st.write(r'''
    Die Ähnlichkeitsfunktion $s$ kann als Ähnlichkeitsmatrix $S$ zusammengefast werden:
''')
st.latex(r'''
    S = \left( \begin{array}{ccc}
        s(r_1, r_1) & \cdots & s(r_1, r_n) \\
        \vdots & \ddots & \vdots \\
        s(r_n, r_1) & \cdots & s(r_n, r_n) 
	\end{array} \right)     
''')

n_bins = 100
hue = np.linspace(0.0, 1.0, n_bins+1)[0:-1]
S = np.zeros((n_bins, n_bins))
for f in range(n_bins):
    for g in range(n_bins):
        S[f, g] = s_fun(hue[f], hue[g])


tab1, tab2, tab3 = st.tabs(['Euklidische Distanz', 'Quadratische Form Distanz', 'Signature Quadratic Form Distanz'])

tab1.latex(r'''
    \lVert X - Y \rVert_2 = \sqrt{ \sum_{i=1}^n \left(X(r_i) - Y(r_i)\right)^2 } 
''')

tab2.latex('''
    QFD_s(X, Y) = \sqrt{ (x-y)^\intercal \cdot S \cdot (x-y) }
''')


tab3.write(r'''
    Sind $X, Y$ zwei Signaturen (nicht Histogramme) und 
    haben deren zugehörigen Gewichtsvektoren $x \in \mathbb{R}^n$ und $y \in \mathbb{R}^m$ unterschiedliche Dimensionen
    muss erst eine Konkatenation $(x|-y)$ definiert werden, um eine Ähnlichkeit bestimmen zu können.
''')
tab3.latex(r'''
    (x|-y) = \left( \begin{array}{c}
        x_1 \\
        \vdots \\
        x_n \\
        y_1 \\
        \vdots \\
        y_m 
	\end{array} \right)
''')

tab3.write('''
    Die Ähnlichkeit wird dann analog zur quadratischen Form Distanz bestimmt:        
''')
tab3.latex(r'''
    SQFD_s(X, Y) = \sqrt{ (x|-y)^\intercal \cdot \tilde{S} \cdot (x|-y) }
''')
tab3.write(r'''
    Sind die Dimensionen $n=m$ und wird für die erweiterte Ähnlichkeitsmatrix $\tilde{S}$ folgende Form verwendet,
    erhält man die quadratischen Form Distanz:    
''')
tab3.latex(r'''
    \tilde{S}
    = \left( \begin{array}{cc}
        S & S \\
        S & S
	\end{array} \right)
    = \left( \begin{array}{cccccc}
        s(r_1, r_1) & \cdots & s(r_1, r_n) & s(r_1, r_1) & \cdots & s(r_1, r_n) \\
        \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
        s(r_n, r_1) & \cdots & s(r_n, r_n) & s(r_n, r_1) & \cdots & s(r_n, r_n) \\
        s(r_1, r_1) & \cdots & s(r_1, r_n) & s(r_1, r_1) & \cdots & s(r_1, r_n) \\
        \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
        s(r_n, r_1) & \cdots & s(r_n, r_n) & s(r_n, r_1) & \cdots & s(r_n, r_n)
	\end{array} \right)     
''')

S_sqfd = np.zeros((n_bins+n_bins, n_bins+n_bins))
for f in range(n_bins):
    for g in range(n_bins):
        S_sqfd[f, g] = s_fun(hue[f], hue[g])
        S_sqfd[n_bins+f, g] = s_fun(hue[f], hue[g])
        S_sqfd[f, n_bins+g] = s_fun(hue[f], hue[g])
        S_sqfd[n_bins+f, n_bins+g] = s_fun(hue[f], hue[g])




#COLOR_WHEEL_NAMES = ['Rot', 'Orange', 'Gelb', 'Limette', 'Grün', 'Türkis', 'Cyan', 'Azurblau', 'Blau', 'Lila', 'Magenta', 'Purpur']



photoset_path = os.path.join('data', 'photoset')


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


hist_type = st.radio('Histogramm definition der Farbe aus dem Farbkreis:', ['als einzelner Bin', 'mittels einer Standardnormalverteilung'], index=1, horizontal=True)

sigma = 1
if hist_type == 'mittels einer Standardnormalverteilung':
    sigma = s_width * st.slider('Varianz', 0.0, 1.0, 1/3)



# def normalize(a):
#     return (a - np.amin(a))/(np.amax(a) - np.amin(a))

for j in range(0, n_cols):
# for j in [2,4,8]:

    st.subheader(COLOR_WHEEL_NAMES[j])

    col_r, col_g, col_b = ImageColor.getcolor(COLOR_VALUES[j], "RGB")
    col_h, col_l, col_s = colorsys.rgb_to_hls(col_r/255, col_g/255, col_b/255)
    # st.write(h, l, s)
    
    
    # mu = 1.0/n_cols * j
    mu = col_h
    

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
    col_hist_peak[mu_i] = 100.0
    col_hist_peak = col_hist_peak.reshape((n_bins, 1))

    col_hist = col_hist_pdf
    if hist_type == 'als einzelner Bin':
        col_hist = col_hist_peak

    bar_width = 1.0/n_bins/2

    similarity_norm = np.zeros(n_photos)
    similarity_qfd = np.zeros(n_photos)
    similarity_sqfd = np.zeros(n_photos)

    for i in range(n_photos):
    
        photo_hist = histogram_data[i,:]
        photo_hist = photo_hist.reshape((n_bins,1))

        # dot product
        # similarity[i] = photo_hist.T @ col_hist / (linalg.norm(photo_hist)*linalg.norm(col_hist))
        
        hist_diff = photo_hist - col_hist

        # euclidean distance
        similarity_norm[i] = linalg.norm(hist_diff)

        # quadratic form distance
        similarity_qfd[i] = math.sqrt(hist_diff.T @ S @ hist_diff)

        # signature quadratic form distanz
        hist_concat = np.concatenate((photo_hist, -col_hist))
        similarity_sqfd[i] = math.sqrt(hist_concat.T @ S_sqfd @ hist_concat)


    sim_sorted_norm = np.argsort(similarity_norm)
    sim_sorted_qfd = np.argsort(similarity_qfd)
    sim_sorted_sqfd = np.argsort(similarity_sqfd)



    col1, col2, col3, col4 = st.columns(4)
    col1.write('Reihenfolge')
    col2.write('**Euklidische Distanz**')
    col3.write('**QFD**')
    col4.write('**SQFD**')

    for i in range(0, 3):
        col1, col2, col3, col4 = st.columns(4)
        cols = [col2, col3, col4]

        col1.write(f'{i+1}')
        for t, sim_sorted in enumerate([sim_sorted_norm, sim_sorted_qfd, sim_sorted_sqfd]):
            
            
            cols[t].image(img_files_list[sim_sorted[i]])
        

    fig = plt.figure(figsize=(14, 2))
    gs = fig.add_gridspec(1,3)
    ax_img = fig.add_subplot(gs[0,0])
    ax_px = fig.add_subplot(gs[0, 1:3])

    i = st.slider('Die Bilder werden nach ihrer Ähnlichkeit absteigend sortiert. Zeige das Histogram zum Bild in der Ähnlichkeitsabfolge zu QFD folgender Nummer:', 1, 10, 1, 1, key=COLOR_WHEEL_NAMES[j]) - 1
    sim_sorted = sim_sorted_qfd
    image = Image.open(img_files_list[sim_sorted[i]])
    ax_img.imshow(image)
    ax_img.set_title(sim_sorted[i])

    photo_hist = histogram_data[sim_sorted[i],:].reshape((n_bins,1))

    f = mu_i
    if hist_type == 'als einzelner Bin':
        ax_px.bar(hue[f], np.amax(photo_hist), width=bar_width, color='#000000') #, color=COLOR_VALUES[j])

    for g in range(0, n_bins):
        
        red, green, blue = colorsys.hls_to_rgb(hue[g], 0.5, 1.0)

        if hist_type == 'als einzelner Bin':
            # ax_px.bar(hue[g]+bar_width, photo_hist[g], width=bar_width, color=(red, green, blue, 0.2))
            ax_px.bar(hue[g]+bar_width, photo_hist[g]*s_fun(hue[f], hue[g]), width=bar_width, color=(red, green, blue))
            ax_px.bar(hue[g]+bar_width, photo_hist[g], width=bar_width, color=(red, green, blue, 0.2))
        else:
            ax_px.bar(hue[g], col_hist[g], width=bar_width, color=COLOR_VALUES[j])#, color=(r, g, b))
            ax_px.bar(hue[g]+bar_width, photo_hist[g], width=bar_width, color=(red, green, blue, 0.4))

    st.pyplot(fig)
    if hist_type == 'als einzelner Bin':
        st.caption(r'''
            Im Hintergrund schwach zu erkennen ist das Histogram des jewiligen Bildes.
            Der schwarze Balken zeigt den Peak bei dem Farbwert der gesuchten Farbe
            und die farbigen Balken zeigen das Histogram des Bildes in Abhängigkeit (multipliziert) zur Ähnlichkeitsfunktion $s$.
        ''')
    else:
        st.caption(r'''
            Im Hintergrund schwach zu erkennen ist das Histogram des jewiligen Bildes.
            Die farbigen Balken im Vordergrund zeigen die Standardnormalverteilung um den Farbwert der gesuchten Farbe.
        ''')

    


    

    continue


    col_hist = [col_hist_pdf, col_hist_peak]
    for t, sim_sorted in enumerate([sim_sorted_norm, sim_sorted_qfd]):
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

            f = mu_i
            for g in range(0, n_bins):
                
                red, green, blue = colorsys.hls_to_rgb(hue[g], 0.5, 1.0)
                if t == 0:
                    ax_px.bar(hue[g], col_hist[t][g], width=bar_width, color=COLOR_VALUES[j])#, color=(r, g, b))

                
                
                if t == 1:
                    ax_px.bar(hue[g]+bar_width, photo_hist[g], width=bar_width, color=(red, green, blue, 0.2))
                    ax_px.bar(hue[g]+bar_width, photo_hist[g]*s_fun(hue[f], hue[g]), width=bar_width, color=(red, green, blue))
                else:
                    ax_px.bar(hue[g]+bar_width, photo_hist[g], width=bar_width, color=(red, green, blue))

            # ax.plot(x, y)

            # tabs[t].pyplot(fig)