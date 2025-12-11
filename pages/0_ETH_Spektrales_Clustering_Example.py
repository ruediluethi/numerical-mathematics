
import os
import streamlit as st
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import colorsys

import pandas as pd

from scipy.ndimage import gaussian_filter

import random
import pandas as pd

from sklearn.cluster import KMeans
import math

COLOR_WHEEL_NAMES = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
n_cols = len(COLOR_WHEEL_NAMES)

photoset_path = os.path.join('data', 'photoset')

img_files_list = []
for file in os.listdir(photoset_path):
    img_path = os.path.join(photoset_path, file)
    if os.path.isfile(img_path):
        img_files_list.append(img_path.replace('\\', '/'))

# st.write(img_files_list)

# img_path = random.choice(img_files_list)

@st.cache_data
def get_weight_matrix(image, sigma_X=1.0, sigma_F=0.1):
    
    data = np.asarray(image)

    R = data[:,:,0]
    G = data[:,:,1]
    B = data[:,:,2]

    img_size = R.shape[0]

    st.write(img_size)

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
    st.write(n)

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

def w_fun(L, y_cap_scale=1.5):
    return np.clip((np.sin(L * np.pi*2 - np.pi/2)/2 + 0.5) * y_cap_scale, 0.0, 1.0)

# for image_file_path in img_files_list[32:33]:
# for image_file_path in img_files_list[6:7]:
# for image_file_path in img_files_list[84:85]:
# for image_file_path in img_files_list[40:41]:
# for image_file_path in img_files_list[0:10]:
# for image_file_path in [img_files_list[6], img_files_list[84]]:
for image_file_path in img_files_list[55:]:


    # image_file_path = img_files_list[39]
    # image = Image.open(img_files_list[45])
    # image = Image.open(img_files_list[48])
    # image = Image.open(img_files_list[34])
    image = Image.open(image_file_path)
    orig_size = (image.width, image.height)

    # Resize image to half of its original size
    new_size = (image.width // 8, image.height // 8)
    # new_size = (image.width // 64, image.height // 64)
    image = image.resize(new_size)

    fig, ax = plt.subplots()
    ax.imshow(image)
    st.pyplot(fig)



    W = get_weight_matrix(image)

    fig, ax = plt.subplots()
    ax.imshow(W, aspect='equal')
    st.pyplot(fig)

    @st.cache_data
    def spectral_clustering(W, k=2, do_plot=False):

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
        if do_plot:
            hist, bin_edges = np.histogram(v_2, bins=20)
            fig, ax = plt.subplots()
            ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
            st.pyplot(fig)

        v_k = V[:, 1:k]

        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(v_k)

        return labels

    calc_layers_progress = st.progress(0.0, 'calc layers ...')
    for k in range(2, 21):
    # for k in range(11, 21):
        calc_layers_progress.progress(k/20, 'calc layers ...')
        # k = 10
        labels = spectral_clustering(W, k=k, do_plot=False)


        data = np.asarray(image)

        R = data[:,:,0]
        G = data[:,:,1]
        B = data[:,:,2]

        mean_cluster_colors = []
        for i in range(k):
            mean_cluster_colors.append({
                'count': 0,
                'R': 0, 
                'G': 0,
                'B': 0,
                'H': np.zeros(0),
                'L': np.zeros(0),
                'S': np.zeros(0),
                'w': np.zeros(0),
                'mask': np.zeros(orig_size)
            })

        img_cluster = np.zeros(new_size)
        for i, label in enumerate(labels):
            x_i = i // new_size[0]
            y_i = i % new_size[0]

            # if labels[i] == 0:
            #     img_cluster[x_i, y_i]
            img_cluster[x_i, y_i] = label

            r = float(R[x_i, y_i])
            g = float(G[x_i, y_i])
            b = float(B[x_i, y_i])
            h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)

            mean_cluster_colors[label]['count'] += 1
            mean_cluster_colors[label]['R'] += r
            mean_cluster_colors[label]['G'] += g
            mean_cluster_colors[label]['B'] += b
            mean_cluster_colors[label]['H'] = np.append(mean_cluster_colors[label]['H'], h)
            mean_cluster_colors[label]['L'] = np.append(mean_cluster_colors[label]['L'], l)
            mean_cluster_colors[label]['S'] = np.append(mean_cluster_colors[label]['S'], s)
            mean_cluster_colors[label]['w'] = np.append(mean_cluster_colors[label]['w'], s**2 * w_fun(l))

            for x_k in range(8):
                for y_k in range(8):
                    mean_cluster_colors[label]['mask'][x_i*8 + x_k, y_i*8 + y_k] = 1.0

        # fig, ax = plt.subplots()
        # ax.imshow(img_cluster, aspect='equal')
        # st.pyplot(fig)

        for i in range(k):
            count = mean_cluster_colors[i]['count']

            r = mean_cluster_colors[i]['R'] / count
            g = mean_cluster_colors[i]['G'] / count
            b = mean_cluster_colors[i]['B'] / count

            h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)

            red, green, blue = colorsys.hls_to_rgb(h, l, 1-(1-(0.2 + 0.8*s)) )


            mean_cluster_colors[i]['R'] = red
            mean_cluster_colors[i]['G'] = green
            mean_cluster_colors[i]['B'] = blue


        # Create a sorted list of indices by 'count' descending
        sorted_indices = sorted(range(len(mean_cluster_colors)), key=lambda i: mean_cluster_colors[i]['count'], reverse=True)
        # st.write(mean_cluster_colors[sorted_indices])

        R_ = np.ones(orig_size)
        G_ = np.ones(orig_size)
        B_ = np.ones(orig_size)

        
        col_names = []
        k_ = 5 + k/10*5
        # if k > 10:
        #     k_ = k

        for i_sort, sort_idx in enumerate(sorted_indices):

            # st.write(mean_cluster_colors[sort_idx])

            calc_colors = False
            if calc_colors:
                count = mean_cluster_colors[sort_idx]['count']

                H = mean_cluster_colors[sort_idx]['H']
                L = mean_cluster_colors[sort_idx]['L']
                S = mean_cluster_colors[sort_idx]['S']

                w = mean_cluster_colors[sort_idx]['w']

                n_bins = 100
                H_hist, H_bins = np.histogram(H, bins=n_bins, range=(0.0, 1.0), density=True, weights=w)
                

                if np.median(S) < 0.1 or np.median(L) < 0.3 or np.median(L) > 0.8:
                    pass

                else:

                    st.write(count, np.median(S), np.median(L))

                    fig_hist, ax_hist = plt.subplots()
                    for j in range(n_bins):
                        r, g, b = colorsys.hls_to_rgb(H_bins[j+1], 0.5, 1.0)
                        ax_hist.bar(H_bins[j+1], H_hist[j], width=np.abs(np.amax(H_bins) - np.amin(H_bins))/n_bins, color=(r, g, b))
                    
                    max_idx = np.argmax(H_hist)
                    r, g, b = colorsys.hls_to_rgb(H_bins[max_idx+1], 0.5, 1.0)
                    ax_hist.plot(H_bins[max_idx+1], H_hist[max_idx], 'o', markersize=12, color='#FFFFFF')
                    ax_hist.plot(H_bins[max_idx+1], H_hist[max_idx], 'o', markersize=8, color=(r, g, b))

                    col_borders = np.array([0.0, 0.083, 0.2, 0.42, 0.58, 0.75, 0.92, 1.0])
                    for j, c in enumerate(col_borders):
                        if j > 0:
                            c_prev = col_borders[j-1]

                            r, g, b = colorsys.hls_to_rgb((c_prev + c)/2, 0.5, 1.0)
                            ax_hist.fill([max(0.0, c_prev), max(0.0, c_prev), min(1.0, c), min(1.0, c)], 
                                            [0.0, np.amax(H_hist), np.amax(H_hist), 0.0], color=(r, g, b), alpha=0.3)
                            
                            if c_prev < H_bins[max_idx+1] and H_bins[max_idx+1] <= c:
                                crnt_col_name = COLOR_WHEEL_NAMES[(j-1)%n_cols]
                                if crnt_col_name not in col_names:
                                    col_names.append(crnt_col_name)

                    
                    st.pyplot(fig_hist)
            

            mask = mean_cluster_colors[sort_idx]['mask']



            blurred_mask = gaussian_filter(mask, sigma=max(1,k_-i_sort+1))

            # Threshold blurred_mask: values < 0.5 to 0, >= 0.5 to 1 (vectorized)
            # blurred_mask = (blurred_mask >= 0.5).astype(float)

            # Threshold blurred_mask: values < 0.5 to 0, >= 0.5 to 1
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if i_sort == 0 or blurred_mask[i, j] > min(0.5, 0.5 * (i_sort+1) / k_):
                    #     blurred_mask[i, j] = 0
                    # else:
                    #     blurred_mask[i, j] = 1
                        
                        R_[i, j] = mean_cluster_colors[sort_idx]['R']
                        G_[i, j] = mean_cluster_colors[sort_idx]['G']
                        B_[i, j] = mean_cluster_colors[sort_idx]['B']

                    # R_[i, j] += mean_cluster_colors[sort_idx]['R']*blurred_mask[i, j]
                    # G_[i, j] += mean_cluster_colors[sort_idx]['G']*blurred_mask[i, j]
                    # B_[i, j] += mean_cluster_colors[sort_idx]['B']*blurred_mask[i, j]

        # fig, ax = plt.subplots()
        # ax.imshow(blurred_mask, aspect='equal')
        # st.pyplot(fig)

        # st.stop()

        # R_ = np.zeros(orig_size)
        # G_ = np.zeros(orig_size)
        # B_ = np.zeros(orig_size)

        # for i, label in enumerate(labels):
        #     x_i = i // new_size[0]
        #     y_i = i % new_size[0]

        #     for x_k in range(8):
        #         for y_k in range(8):
        #             R_[x_i*8 + x_k, y_i*8 + y_k] = mean_cluster_colors[label]['R']
        #             G_[x_i*8 + x_k, y_i*8 + y_k] = mean_cluster_colors[label]['G']
        #             B_[x_i*8 + x_k, y_i*8 + y_k] = mean_cluster_colors[label]['B']

        #     # R_[x_i, y_i] = mean_cluster_colors[label]['R']
        #     # G_[x_i, y_i] = mean_cluster_colors[label]['G']
        #     # B_[x_i, y_i] = mean_cluster_colors[label]['B']

        R_ = (R_*255).astype(np.uint8)
        G_ = (G_*255).astype(np.uint8)
        B_ = (B_*255).astype(np.uint8)

        # st.write(type(R))



        original = np.dstack((R_,G_,B_))

        image_original = Image.fromarray(original)
        # st.image(image_original)
        # fig, ax = plt.subplots()
        # ax.imshow(image_original)
        # st.pyplot(fig)

        img_file_basename = os.path.basename(image_file_path)
        name, ext = os.path.splitext(img_file_basename)
        new_filename = f"{name}_{k}{ext}"
        output_path = os.path.join('data', 'clustered_photos', new_filename)

        image_original.save(output_path)

    calc_layers_progress.empty()

    st.write(img_file_basename, col_names)

    # Save or update CSV with filename and colors
    
    csv_path = os.path.join('data', 'image_colors.csv')
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
