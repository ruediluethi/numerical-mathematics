import math
import os
import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px

st.title('Transformation zwischen Koordinatensystemen')

path_to_data = os.path.join('data', 'zylinderrand')
data = np.genfromtxt(os.path.join(path_to_data, '1612262607900-Log.txt'), delimiter=',')

n = min(2000,data.size)

MKS = np.zeros([4,n])
WKS = np.zeros([4,n])
bm = np.zeros(n)
v = np.zeros(n)
a = np.zeros(n)
c = np.zeros(n)

k = 0
for i in range(100,n,1):
    row_prev = data[i-1]
    pos_m_prev = np.array([row_prev[0], row_prev[1], row_prev[2]]).reshape(3,1)

    row = data[i]

    x_m = row[0]
    y_m = row[1]
    z_m = row[2]
    pos_m_i = np.array([x_m, y_m, z_m]).reshape(3,1)
    a_m = row[3]
    c_m = row[4]

    bm_i = row[5]
    ax_i = row[6]
    tr_i = row[7]

    x_w = row[8]
    y_w = row[9]
    z_w = row[10]
    a_w = row[11]
    c_w = row[12]

    t_i = row[13]

    v[i] = np.linalg.norm(pos_m_i - pos_m_prev)

    # if z_m > 0 or z_m < -30:
    #     continue
    # if bm_i < 0.3:
    #     continue

    MKS[:,k] = np.array([x_m, y_m, z_m, 1]).reshape(4,)
    WKS[:,k] = np.array([x_w, y_w, z_w, 1]).reshape(4,)
    bm[k] = bm_i
    a[k] = a_w
    c[k] = c_w

    k = k+1

st.write(MKS)
st.write(bm)

st.line_chart(v)

st.subheader('Maschinen-Koordinatensystem (global)')
st.plotly_chart(px.scatter_3d(pd.DataFrame({
    'bm': bm[0:k],
    'x': MKS[0,0:k],
    'y': MKS[1,0:k],
    'z': MKS[2,0:k]
}), x='x', y='y', z='z',
    color='bm',
    color_continuous_scale=px.colors.sequential.Plasma,
    # range_x=[-100,100],
    # range_y=[-100,100]
))

st.subheader('Werkstück-Koordinatensystem (lokal)')
st.plotly_chart(px.scatter_3d(pd.DataFrame({
    'bm': bm[0:k],
    'x': WKS[0,0:k],
    'y': WKS[1,0:k],
    'z': WKS[2,0:k]
}), x='x', y='y', z='z',
    color='bm',
    color_continuous_scale=px.colors.sequential.Plasma,
))



debug_res = 16
debug_r = 100

A_axis = np.zeros([4,debug_res +2])
A_axis[:,0] = np.array([ debug_r*5 , 0 , 0 , 1 ]).reshape(4,)
A_axis[:,1] = np.array([ 0 , 0 , 0 , 1 ]).reshape(4,)
C_axis = np.zeros([4,debug_res +2])
C_axis[:,0] = np.array([ 0 , 0 , debug_r*5 , 1 ]).reshape(4,)
C_axis[:,1] = np.array([ 0 , 0 , 0 , 1 ]).reshape(4,)
for i in range(0,debug_res):
    alpha = 2*math.pi * i/debug_res
    cos_alpha = math.cos(alpha)*debug_r
    sin_alpha = math.sin(alpha)*debug_r
    A_axis[:,2+i] = np.array([ 0 , cos_alpha , sin_alpha , 1 ]).reshape(4,)
    C_axis[:,2+i] = np.array([ cos_alpha , sin_alpha , 0 , 1 ]).reshape(4,)

def plot_3d_lines(lines):

    n = 0
    for line in lines:
        n = n + line.shape[1]

    L = np.zeros([4,n])
    colors = np.zeros([1,n])
    k = 0
    for line in lines:
        L[:,k:k+line.shape[1]] = line
        colors[:,k:k+line.shape[1]] = np.ones([1,line.shape[1]])*k
        k = k + line.shape[1]

    st.plotly_chart(px.line_3d(pd.DataFrame({
        'x': L[0,:],
        'y': -L[1,:],
        'z': L[2,:],
        'color': colors.flatten()
    }), x='x', y='y', z='z', color='color'))

    st.write(colors)


t_A = np.array([[0, -310-70, -465-120-70]])
T_A = np.eye(4)
T_A[0:3,3] = t_A.reshape(3,)
T_A_inv = np.eye(4)
T_A_inv[0:3,3] = t_A.reshape(3,)*-1

t_C = np.array([[380, 70, 0]])
T_C = np.eye(4)
T_C[0:3,3] = t_C.reshape(3,)
T_C_inv = np.eye(4)
T_C_inv[0:3,3] = t_C.reshape(3,)*-1

R_A = np.eye(4)
R_C = np.eye(4)

i_end = st.slider('step', 1, n, 100, 1)

TCP_global = np.zeros([4, i_end])

for i in range(0,i_end):

    alpha = -a[i]/360 * 2*math.pi
    gamma = -c[i]/360 * 2*math.pi

    R_A = np.array([
        [ 1, 0, 0, 0 ],
        [ 0, math.cos(alpha), -math.sin(alpha), 0 ],
        [ 0, math.sin(alpha),  math.cos(alpha), 0 ],
        [ 0, 0, 0, 1 ]
    ])

    R_C = np.array([
        [ math.cos(gamma), -math.sin(gamma), 0, 0 ],
        [ math.sin(gamma),  math.cos(gamma), 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
    ])

    WKS_trafo = T_A @ R_A @ T_C @ R_C @ T_C_inv @ T_A_inv @ WKS
    TCP_global[:,i] = WKS_trafo[:,i]

A_axis_trafo = T_A @ R_A @ A_axis
C_axis_trafo = T_A @ R_A @ T_C @ R_C @ C_axis


plot_3d_lines([A_axis_trafo, C_axis_trafo, WKS_trafo[:,0:i], TCP_global])

# fig = px.line_3d(pd.DataFrame({
#     'x': A_axis[0,:],
#     'y': A_axis[1,:],
#     'z': A_axis[2,:]
# }), x='x', y='y', z='z')

# st.plotly_chart(fig)



# st.header('G550 5-Achs-Universalmaschine')

# path_to_data = os.path.join('data', 'measurement_20210818T153046304_27')
# path_to_file = os.path.join(path_to_data, 'measurement_20210818T153046304_27_axis_2-2207_1.csv')
# df = pd.read_csv(path_to_file, skiprows=[1])

# st.write(df.size)
# n = df.size
# timestamps = np.zeros(n)
# for index, row in df.iterrows():
#     timestamps[index] = pd.Timestamp(row['time']).timestamp()

# timestamps = timestamps - timestamps[0]

# fields = df['field'].to_numpy()
# values = df['value'].to_numpy()

# A_indices = np.where(fields == 'A')
# B_indices = np.where(fields == 'B')

# fig, ax = plt.subplots()
# ax.plot(np.take(timestamps, A_indices).T, np.take(values, A_indices).T)
# ax.plot(np.take(timestamps, B_indices).T, np.take(values, B_indices).T)
# st.pyplot(fig)
