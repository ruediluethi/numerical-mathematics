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
st.write('''
    Anhand des im vorderen Schritt berechneten Histogramms wird zu jedem Bild ermittelt ob eine der 6 Grundfarben Rot, Gelb, Grün, Cyan, Blau oder Magenta in dem Bild vorkommt.
    Die in dem Bild enthalten Farben werden als Merkmalsliste gespeichert.  
''')

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
show_all_images = st.checkbox('Alle Bilder anzeigen')
for i, img_file in enumerate(img_files_list):

    show_images = True
    if i >= 15 and show_all_images == False:
        show_images = False

    if i%5 == 0 and show_images:
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        rand_hist_i = random.randrange(5)
        hist_container = st.container()

    show_hist_plot = False
    if i%5 == rand_hist_i and show_images:
        if show_all_hist_plots:
            show_hist_plot = True


    crnt_column = cols[i%5]
    if show_images:
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
    if show_images:
        crnt_column.write(caption, unsafe_allow_html=True)


st.write('''
    Die vorkommenden Farben und deren Kombinationen werden gezählt.      
''')

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
    fig, ax = plt.subplots(figsize=(9, 3))
    
    count_summary = []
    bar_counter = 0
    for k, i in enumerate(a):
        

        count = 0
        for j in b:
            if a_is_in_b(i, j):
                count += 1
        
        count_summary.append(count)

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

    ax.set_xlabel('Farbkombinationen')
    ax.set_ylabel('Anzahl')

    st.pyplot(fig)

    return count_summary


all_count_summaries = []
all_combinations = []    

base_colors = count_a_in_b(lk, data)
st.caption('''
    Welche Farbe kommt wie oft in den Bildern vor?       
''')
all_count_summaries.append(base_colors)
all_combinations.append(lk)


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
    
    count_summary = count_a_in_b(lk, data)
    all_count_summaries.append(count_summary)
    all_combinations.append(lk)

    if k == 0:
        st.caption('''
            Die Farbkombination Gelb und Blau scheint sehr häufig zu sein.
        ''')

    if k == 1:
        st.caption('''
            Auch Rot, Gelb, Blau scheint häufig zu sein.
        ''')

st.header('Wahrscheinlichkeiten')


st.subheader('Support')
st.write(r'''
    Sei $X$ das Ereignis, dass eine Farbe in einem Bild vorkommt und sei $Y$ eine andere Farbe.
    So ist $\mathbb{P}(X \cap Y)$ die Wahrscheinlichkeit, dass beide Farben in einem Bild in der Datenbank $D$ vorkommen:
''')
st.latex(r'''
    \mathbb{P}(X \cap Y) = \frac{|X \cap Y|}{|D|}
''')


st.write('**Beispiel**')


example_cols = st.multiselect('Auswahl der Farben für eine Beispielberechnung', COLOR_WHEEL_NAMES, default=['Rot', 'Gelb', 'Blau'])

# st.write(example_cols)
example_col_ids = []
for col_name in example_cols:
    example_col_ids.append(COLOR_WHEEL_NAMES.index(col_name))
example_col_ids.sort()
# st.write(example_col_ids)

st.write(f'$|D| = {len(img_files_list)}$ Bilder')

formula_names = ['X', 'Y', 'Z', 'W', 'V', 'U']

used_formula_names = []
for j, col_j in enumerate(example_col_ids):
    st.write(f'${formula_names[j]} = $ {COLOR_WHEEL_NAMES[col_j]}, $|{formula_names[j]}| = {base_colors[col_j]}$')
    used_formula_names.append(formula_names[j])


def get_count(example_col_ids):

    example_comb = all_combinations[len(example_col_ids)-1]
    example_count = all_count_summaries[len(example_col_ids)-1]

    for i, col_comb in enumerate(example_comb):
        comb_found = True
        for j, col_j in enumerate(example_col_ids):
            if col_comb[j] != col_j:
                comb_found = False
                break
        if comb_found:
            return example_count[i]

example_count = get_count(example_col_ids)

# st.write(col_comb)
# st.write(example_count[i])
cap_string = ' \cap '.join(used_formula_names)
st.write(f"$|{cap_string}| = {example_count}$")
P_cap = example_count / len(img_files_list)

P_string = r'\mathbb{P}'
st.write(f'${P_string}({cap_string}) = {round(P_cap*100, 2)}$%')

st.info(f'*Die Wahrscheinlichkeit, dass die Farben **{', '.join(example_cols)}** in einem Bild vorkommen beträgt **{round(P_cap*100, 2)}%**.*')

st.subheader('Konfidenz')
st.write(r'''
    Wird eine Farbe $Y$ durch eine andere Farbe $X$ impliziert $X \Rightarrow Y$? 
    Wie wahrscheinlich ist es, dass wenn eine Farbe $X$ im Bild enthalten ist, dass auch die Farbe $Y$ enthalten ist?
    Dies wird durch die bedingte Wahrscheinlichkeit $\mathbb{P}(Y | X)$ beschrieben:
''')
st.latex(r'''
    X \Rightarrow Y: \quad \mathbb{P}(Y | X) = \frac{\mathbb{P}(X \cap Y)}{\mathbb{P}(X)}
''')

st.write('**Beispiel**')

col1, col2 = st.columns(2)
X = col1.selectbox('X', COLOR_WHEEL_NAMES, index=4)
Y = col2.selectbox('Y', COLOR_WHEEL_NAMES, index=1)
X_id = COLOR_WHEEL_NAMES.index(X)
Y_id = COLOR_WHEEL_NAMES.index(Y)

if X_id == Y_id:
    st.warning('Bitte zwei unterschiedliche Farben auswählen!')
else:

    if X_id < Y_id:
        XY_cap = get_count([X_id, Y_id])
    else:
        XY_cap = get_count([Y_id, X_id])

    P_X = base_colors[X_id] / len(img_files_list)
    P_X_cap_Y = XY_cap / len(img_files_list)

    st.write(f'$X = $ {COLOR_WHEEL_NAMES[X_id]}, $Y = $ {COLOR_WHEEL_NAMES[Y_id]}')
    st.write(f'$|X| = {base_colors[X_id]}$, ${r'\mathbb{P}'}(X) = {round(P_X*100, 2)}$%')
    st.write(f'$|X \cap Y| = {XY_cap}$, ${r'\mathbb{P}'}(X \cap Y) = {round(P_X_cap_Y*100, 2)}$%')
    P_Y_pipe_X = P_X_cap_Y / P_X
    st.write(f'$X \Rightarrow Y: \quad {r'\mathbb{P}'}(Y | X) = {round(P_Y_pipe_X * 100, 2)}$%')

    st.info(f'*Ist die Farbe **{X}** im Bild enthalten, so ist zu **{round(P_Y_pipe_X * 100, 2)}%** auch **{Y}** zu sehen.*')
