import os
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import colorsys

from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

import pandas as pd

st.title('Spektrales Clustering')

examples = ['Ähnliche Farbflächen in einem Bild', '2D Punktewolke mit Distanzfunktion']
example_selection = st.radio('Auswahl eines Beispieles', examples)

tab_img = example_selection == examples[0]
tab_points = example_selection == examples[1]

some_images = ['05.png', '23.png', '41.png', '45.png', '73.png', '49.png']#, '09.png']
COLOR_WHEEL_NAMES = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
n_cols = len(COLOR_WHEEL_NAMES)

if tab_img:
    selected_idx = 3
    if 'selected_idx' in st.session_state:
        selected_idx = st.session_state['selected_idx']

    cols = st.columns(len(some_images))
    for i, col in enumerate(cols):
        with col:
            img_path = os.path.join('data', 'photoset', some_images[i])
            st.image(img_path)
            if i == selected_idx:
                pass
            else:
                if st.button(f"auswählen", key=f"select_img_{i}"):
                    selected_idx = i
                    st.session_state['selected_idx'] = i
                    st.rerun()
                    st.stop()

    img_file_basename = some_images[selected_idx]
    img_path = os.path.join('data', 'photoset', img_file_basename)


    image_orig = Image.open(img_path)
    orig_size = image_orig.width
    new_size = st.select_slider('Das Originalbild auf folgende Größe skalieren', options=[2**i for i in range(1, 6)], value=16)
    image = image_orig.resize((new_size,new_size))
    img_size = image.width

    new_to_orig_scale = orig_size // new_size

if tab_points:

    n = st.slider('Anzahl Punkte', 2, 300, 150)
    
    
    r1 = st.slider('Radius innen', 1, 100, 20)
    r2 = st.slider('Radius außen', 1, 100, 60)

    n_inside = n * r1 // (r1 + r2)
    n_outside = n - n_inside

    angle_range = st.slider('Winkelbereich', 0.0, np.pi*2, np.pi*2)

    @st.cache_data
    def get_random_points(n_inside, n_outside, r1, r2, angle_range):
        n = n_inside + n_outside
        d = 2
        X = np.zeros((n,d))
        for i in range(n):
            if i < n_inside:
                r = r1
            else:
                r = r2

            angle = np.random.rand() * angle_range
            radius = np.random.randn()*3 + r

            X[i,0] = radius * np.cos(angle)
            X[i,1] = radius * np.sin(angle)

        return X
    
    X = get_random_points(n_inside, n_outside, r1, r2, angle_range)

if tab_img:
    st.write(r'''
        Gegeben sei ein Bild mit $\sqrt{n} \times \sqrt{n} = n$ Pixeln.
        Diese Pixel können sowohl durch ihre Farbe (RGB oder HSL) $f_i \in \mathbb{R}^3$
        als auch durch ihre Position im Bild $x_i \in \mathbb{R}^2$ beschrieben werden.
        Damit wird ein gewichteter Graph $G=(V,E,W)$ mittels eines Kernels definiert
        und als Adjazenzmatrix $W$ dargestellt:
    ''')
    st.latex(r'''
        w_{ij} = \exp \left( - \frac{\left\Vert x_i - x_j \right\Vert^2}{\sigma_X^2} - \frac{\left\Vert f_i - f_j \right\Vert^2}{\sigma_F^2} \right)
    ''')


if tab_points:
    st.write(r'''
        Seien $n$ Datenpunkte der Dimension $d=2$ als Matrix $X \in \mathbb{R}^{n \times d}$ gegeben.
        Damit wird ein gewichteter Graph $G=(V,E,W)$ mittels eines Kernels definiert
        und als Adjazenzmatrix $W$ dargestellt:
    ''')
    st.latex(r'''
        w_{ij} = \exp \left( - \frac{\left\Vert x_i - x_j \right\Vert^2}{2\varepsilon^2} \right)
    ''')

        
if tab_img:
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image_orig, alpha=0.8)
    for i in range(new_size):
        for j in range(new_size):
            (r,g,b) = image.getpixel((i,j))
            # st.write(color)
            ax[0].plot(i * (orig_size/new_size) + (orig_size/(2*new_size)),
                    j * (orig_size/new_size) + (orig_size/(2*new_size)),
                    'o', markersize=new_to_orig_scale*0.2, color=(r/255, g/255, b/255))
                    # markeredgecolor='black', markeredgewidth=0.1)

    @st.cache_data
    def get_weight_matrix(image, sigma_X=0.2, sigma_F=0.1):
        
        data = np.asarray(image)

        R = data[:,:,0]
        G = data[:,:,1]
        B = data[:,:,2]

        img_size = R.shape[0]

        H = np.zeros((img_size, img_size))
        L = np.zeros((img_size, img_size))
        S = np.zeros((img_size, img_size))
        for i in range(img_size):
            for j in range(img_size):
                r, g, b = R[i,j], G[i,j], B[i,j]
                h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
                H[i,j] = h
                L[i,j] = l
                S[i,j] = s

        # st.write(H)

        n = R.shape[0] * R.shape[1]

        W = np.zeros((n, n))

        calc_progress = st.progress(0.0, 'calc weight matrix ...')
        for i in range(n):
            calc_progress.progress(i/n, 'calc weight matrix ...')
            x_i = i // img_size
            y_i = i % img_size

            RGB_i = np.array([R[x_i, y_i], G[x_i, y_i], B[x_i, y_i]]) / 255.0
            # H_i = H[x_i, y_i]
            # L_i = L[x_i, y_i]
            F_i = np.array([H[x_i, y_i], 0.4*L[x_i, y_i]])#, S[x_i, y_i]])
            X_i = np.array([x_i, y_i]) / img_size

            for j in range(i):
                x_j = j // img_size
                y_j = j % img_size

                # dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                # if dist > 0.2:
                #     continue

                RGB_j = np.array([R[x_j, y_j], G[x_j, y_j], B[x_j, y_j]]) / 255.0
                # H_j = H[x_j, y_j]
                # L_j = L[x_j, y_j]
                F_j = np.array([H[x_j, y_j], 0.4*L[x_j, y_j]])#, S[x_j, y_j]])
                X_j = np.array([x_j / img_size, y_j / img_size])

                W[i, j] = np.exp(
                            # -np.linalg.norm(RGB_i - RGB_j)**2 / (sigma_F**2)
                            -np.linalg.norm(F_i - F_j)**2 / (sigma_F**2)
                            # -0.2*np.linalg.norm(H_i - H_j)**2 / (sigma_F**2)
                            -np.linalg.norm(X_i - X_j)**2 / (sigma_X**2)
                        )


        calc_progress.empty()

        W = W + W.T

        return W

    sigma_X = st.slider(r'$\sigma_X$: Gewichtung der räumlichen Distanz', min_value=0.01, max_value=1.0, value=0.5)
    sigma_F = st.slider(r'$\sigma_F$: Gewichtung des Farbraums in HSL transformiert', min_value=0.01, max_value=1.0, value=0.2)

    W = get_weight_matrix(image, sigma_X=sigma_X, sigma_F=sigma_F)

    ax[1].imshow(W, aspect='equal')
    st.pyplot(fig)



@st.cache_data
def spectral_clustering(W, k=2):

    n = W.shape[0]


    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,i] += W[i,j]

    # st.write(D)

    L = D - W

    D_ = np.zeros((n,n))
    for i in range(n):
        D_[i,i] = D[i,i]**(-1/2)

    # st.write(D_)

    L_ = D_ @ L @ D_

    lambdas, V = np.linalg.eig(L_)



    v_2 = V[:,1:2]
    

    v_k = V[:, 1:k]

    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(v_k)

    return labels, v_2


if tab_points:

    distance_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i,j] = np.linalg.norm(X[i,:] - X[j,:])

    # max_dist = np.amax(distance_matrix)
    # min_dist = np.amin(distance_matrix + np.eye(n)*max_dist)
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    mean_dist = distance_matrix[mask].mean()
    max_dist = distance_matrix[mask].max()
    min_dist = distance_matrix[mask].min()

    epsilon_estimation = (float)(min(r1, r2))*0.5
    if st.button('Epsilon schätzen'):
        n_eps = 10
        silhouette_trend = -np.ones(n_eps)
        entropy_trend = -np.ones(n_eps)
        epsilon_list = np.linspace(mean_dist*0.1, mean_dist*0.3, n_eps)

        progress = st.progress(0)
        for i_eps, epsilon in enumerate(epsilon_list):
            progress.progress(i_eps/n_eps)
            
            W_ = np.exp( - distance_matrix**2 / (2*epsilon**2)) - np.eye(n)


            labels, v_2 = spectral_clustering(W_, k=2)

            idx_a = np.argwhere(labels == 0).flatten()
            idx_b = np.argwhere(labels == 1).flatten()

            entropy = - (idx_a.size/n) * np.log2(idx_a.size/n) - (idx_b.size/n) * np.log2(idx_b.size/n)

            # silhouette score
            if idx_a.size > 1 and idx_b.size > 1:
                
                s_sum = 0
                for i in range(n):

                    C_i = idx_a if i in idx_a else idx_b
                    C = idx_b if i in idx_a else idx_a

                    a_i_sum = 0
                    # summe aller distanzen zu punkten im eigenen cluster
                    for j in C_i:
                        a_i_sum += np.linalg.norm(v_2[i] - v_2[j])
                    a_i = 1/(C_i.size - 1) * a_i_sum

                    b_i_sum = 0
                    # summe aller distanzen zu punkten im anderen cluster
                    for j in C:
                        b_i_sum += np.linalg.norm(v_2[i] - v_2[j])
                    b_i = 1/(C.size - 1) * b_i_sum
                    
                    s_i = (b_i - a_i) / max(a_i, b_i)
                    s_sum += s_i

                silhouette_score = 1/n * s_sum
            else:
                silhouette_score = 0

            silhouette_trend[i_eps] = silhouette_score
            entropy_trend[i_eps] = entropy
        progress.empty()

        fig, ax = plt.subplots()
        ax.plot(epsilon_list, silhouette_trend, '-o', label='Silhouette Score')
        ax.plot(epsilon_list, entropy_trend, '-o', label='Entropy')
        ax.plot(epsilon_list, silhouette_trend * entropy_trend, '-o', label='Product')
        ax.legend()
        st.pyplot(fig)

        epsilon_estimation = epsilon_list[np.argmax(entropy_trend * silhouette_trend)]


    # epsilon = st.slider(r'$\varepsilon$', min_dist, max_dist, mean_dist)
    epsilon = st.slider(r'$\varepsilon$', 1.0, (float)(max(r1, r2)), epsilon_estimation)
    

    W = np.exp( - distance_matrix**2 / (2*epsilon**2)) - np.eye(n)


    fig, ax = plt.subplots(1,2)
    ax[0].plot(X[:,0], X[:,1], '.k')
    ax[0].set_aspect('equal')
    ax[1].imshow(W, aspect='equal')
    st.pyplot(fig)

st.write(r'''
    Sei weiter die Diagonalmatrix $D$ als *degree matrix* (Summe aller Kantengewichte eines Knoten) definiert:      
''')
st.latex(r'''
    d_{ii} = \sum_{j} w_{ij}
''')
st.write(r'''
    Und sei der *Laplacian* vom Graph $G$ definiert als      
''')
st.latex(r'''
    L = D - W
''')

st.write(r'''
    Durch die spezielle Form von $L$ gilt für den ersten Eigenvektor $v_1 = \mathbb{1}$     
''')
st.latex(r'''
    L v_1 = \left( D - W \right)  \left( \begin{array}{c}
        1 \\
        1 \\
        \vdots \\
        1
    \end{array} \right) = \lambda_1 v_1 = 0 \\
    \Rightarrow \quad \forall i: \underbrace{\sum_{j} d_{ij}}_{= \underbrace{d_{ii}}_{\sum_{j} w_{ij}}} - \sum_{j} w_{ij} = 0
''')
st.write(r'''
    Also ist der kleinste Eigenwert $\lambda_1 = 0$ (weil $L$ positiv semidefinit).
''')

with st.expander('Notizen'):

    st.write(r'''
        Weiter gilt nach dem Rayleigh-Quotienten für den kleinsten Eigenwert:
    ''')
    st.latex(r'''
        \lambda_1 = \min_{x \neq 0} \frac{x^T L x}{x^T x} = 0
    ''')

    st.write(r'''
        Und es gilt allgemein:
    ''')
    st.latex(r'''
        x^T L x = \frac{1}{2} \sum_{i,j} w_{ij} (x_i - x_j)^2
    ''')
    st.write(r'''
        Herleitung:
    ''')
    st.latex(r'''
        x^T L x = x^T (D - W) x = \sum_{i} d_{ii} x_i^2 - \sum_{i,j} w_{ij} x_i x_j \\
        = \sum_{i,j} w_{ij} x_i^2 - \sum_{i,j} w_{ij} x_i x_j = \frac{1}{2} \sum_{i,j} w_{ij} (x_i^2 - 2 x_i x_j + x_j^2) = \frac{1}{2} \sum_{i,j} w_{ij} (x_i - x_j)^2
    ''')

st.write(r'''
    Normalisieren      
''')
st.latex(r'''
    \tilde{L} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}}      
''')

if tab_img:
    n_clusters = st.slider('Anzahl Cluster', min_value=2, max_value=10, value=3)
if tab_points:
    n_clusters = 2

p = st.container()
fig, ax = plt.subplots(1,2)

labels, v_2 = spectral_clustering(W, k=n_clusters)

hist, bin_edges = np.histogram(v_2, bins=20)
# fig, ax = plt.subplots()
ax[0].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
# st.pyplot(fig)


if tab_points:
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        ax[1].plot(cluster_points[:,0], cluster_points[:,1], '.', label=f'Cluster {i+1}')
    ax[1].set_aspect('equal')
    ax[1].legend()

# st.pyplot(fig)
# st.stop()

def w_fun(L, y_cap_scale=1.5):
    return np.clip((np.sin(L * np.pi*2 - np.pi/2)/2 + 0.5) * y_cap_scale, 0.0, 1.0)

if tab_img:

    data = np.asarray(image)

    R = data[:,:,0]
    G = data[:,:,1]
    B = data[:,:,2]

    mean_cluster_colors = []
    for i in range(n_clusters):
        mean_cluster_colors.append({
            'count': 0,
            'R': 0, 
            'G': 0,
            'B': 0,
            'H': np.array([]),
            'L': np.array([]),
            'S': np.array([]),
            'w': np.array([]),
            'mask': np.zeros((orig_size, orig_size))
        })

    img_cluster = np.zeros((new_size, new_size))
    for i, label in enumerate(labels):
        x_i = i // new_size
        y_i = i % new_size

        img_cluster[x_i, y_i] = label

        r = float(R[x_i, y_i])
        g = float(G[x_i, y_i])
        b = float(B[x_i, y_i])

        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)

        mean_cluster_colors[label]['count'] += 1
        mean_cluster_colors[label]['R'] += r
        mean_cluster_colors[label]['G'] += g
        mean_cluster_colors[label]['B'] += b
        mean_cluster_colors[label]['H'] = np.append(mean_cluster_colors[label]['H'], h)
        mean_cluster_colors[label]['L'] = np.append(mean_cluster_colors[label]['L'], l)
        mean_cluster_colors[label]['S'] = np.append(mean_cluster_colors[label]['S'], s)
        mean_cluster_colors[label]['w'] = np.append(mean_cluster_colors[label]['w'], s**2 * w_fun(l))

        for x_k in range(new_to_orig_scale):
            for y_k in range(new_to_orig_scale):
                mean_cluster_colors[label]['mask'][x_i*new_to_orig_scale + x_k, y_i*new_to_orig_scale + y_k] = 1.0

    col_names = []

    fig_hist, ax_hist = plt.subplots(n_clusters, 1)

    for i in range(n_clusters):
        count = mean_cluster_colors[i]['count']

        # mean_cluster_colors[i]['R'] /= count
        # mean_cluster_colors[i]['G'] /= count
        # mean_cluster_colors[i]['B'] /= count

        r = mean_cluster_colors[i]['R'] / count
        g = mean_cluster_colors[i]['G'] / count
        b = mean_cluster_colors[i]['B'] / count

        # h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)

        # red, green, blue = colorsys.hls_to_rgb(h, l, 1-(1-(0.2 + 0.8*s))**2 )
        # red, green, blue = colorsys.hls_to_rgb(h, l, 1.0 )


        # mean_cluster_colors[i]['R'] = red*255
        # mean_cluster_colors[i]['G'] = green*255
        # mean_cluster_colors[i]['B'] = blue*255

        H = mean_cluster_colors[i]['H']
        L = mean_cluster_colors[i]['L']
        S = mean_cluster_colors[i]['S']
        w = mean_cluster_colors[i]['w']

        n_bins = 100
        H_hist, H_bins = np.histogram(H, bins=n_bins, range=(0.0, 1.0), density=True, weights=w)

        s = 1-(1-(0.2 + 0.8*np.median(S)))**2
        l = np.median(L)
        
        for j in range(n_bins):
            # r, g, b = colorsys.hls_to_rgb(H_bins[j+1], 0.5, 1.0)
            r, g, b = colorsys.hls_to_rgb(H_bins[j+1], l, s)
            ax_hist[i].bar(H_bins[j+1], H_hist[j], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b))
        
        max_idx = np.argmax(H_hist)
        r, g, b = colorsys.hls_to_rgb(H_bins[max_idx+1], l, s)
        ax_hist[i].plot(H_bins[max_idx+1], H_hist[max_idx], 'o', markersize=12, color='#FFFFFF')
        ax_hist[i].plot(H_bins[max_idx+1], H_hist[max_idx], 'o', markersize=8, color=(r, g, b))

        mean_cluster_colors[i]['R'] = r*255
        mean_cluster_colors[i]['G'] = g*255
        mean_cluster_colors[i]['B'] = b*255

        if np.median(S) < 0.1 or l < 0.2 or l > 0.8:
            continue

        col_borders = np.array([0.0, 0.083, 0.2, 0.42, 0.58, 0.75, 0.92, 1.0])
        for j, c in enumerate(col_borders):
            if j > 0:
                c_prev = col_borders[j-1]

                r, g, b = colorsys.hls_to_rgb((c_prev + c)/2, 0.5, 1.0)
                ax_hist[i].fill([max(0.0, c_prev), max(0.0, c_prev), min(1.0, c), min(1.0, c)], 
                                [0.0, np.amax(H_hist), np.amax(H_hist), 0.0], color=(r, g, b), alpha=0.3)
                
                if c_prev < H_bins[max_idx+1] and H_bins[max_idx+1] <= c:
                    crnt_col_name = COLOR_WHEEL_NAMES[(j-1)%n_cols]
                    if crnt_col_name not in col_names:
                        col_names.append(crnt_col_name)

        

    st.pyplot(fig_hist)
                

    # st.write(mean_cluster_colors)
    st.write(col_names)
    csv_path = os.path.join('data', 'photoset', 'features', 'image_colors.csv')
    colors_str = '/'.join(col_names)
    # Try to read existing CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Check if filename already exists
        if (df['filename'] == img_file_basename).any():
            df.loc[df['filename'] == img_file_basename, 'colors'] = colors_str
        else:
            df = pd.concat([df, pd.DataFrame({'filename': [img_file_basename], 'colors': [colors_str]})], ignore_index=True)
    else:
        df = pd.DataFrame({'filename': [img_file_basename], 'colors': [colors_str]})
    df.to_csv(csv_path, index=False)

    sorted_indices = sorted(range(len(mean_cluster_colors)), key=lambda i: mean_cluster_colors[i]['count'], reverse=True)

    R_ = np.ones((orig_size, orig_size))
    G_ = np.ones((orig_size, orig_size))
    B_ = np.ones((orig_size, orig_size))

    for i_sort, sort_idx in enumerate(sorted_indices):
        mask = mean_cluster_colors[sort_idx]['mask']

        blurred_mask = gaussian_filter(mask, sigma=new_to_orig_scale//4)
        # blurred_mask = mask

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if i_sort == 0 or blurred_mask[i, j] > 0.5:
                #     blurred_mask[i, j] = 0
                # else:
                #     blurred_mask[i, j] = 1
                    
                    R_[i, j] = mean_cluster_colors[sort_idx]['R']
                    G_[i, j] = mean_cluster_colors[sort_idx]['G']
                    B_[i, j] = mean_cluster_colors[sort_idx]['B']

    # fig, ax = plt.subplots()
    # ax.imshow(img_cluster, aspect='equal')
    # st.pyplot(fig)

    R_ = R_.astype(np.uint8)
    G_ = G_.astype(np.uint8)
    B_ = B_.astype(np.uint8)

    clustered = np.dstack((R_,G_,B_))
    image_clustered = Image.fromarray(clustered)

    # fig, ax = plt.subplots()
    ax[1].imshow(image_clustered)

p.pyplot(fig)