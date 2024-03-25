import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt



raw_file = os.path.join('data', 'measurement_2022-01-18T080000000_1_0xA0F5.csv')

df = pd.read_csv(raw_file, skiprows=[1])
df.describe()


bmX_all = df['Bm X'].to_numpy()
bmY_all = df['Bm Y'].to_numpy()

t_all = np.array([])
for index, row in df.iterrows():
    t_all = np.append(t_all, pd.Timestamp(row['time']).timestamp() * 1000)


n = st.slider('n_part', 1, 3000, 500, 1)
start_i = st.slider('start_i', 0, t_all.size, 16346, 1)
end_i = start_i + n

duration = t_all[end_i] - t_all[start_i]

t = t_all[start_i:end_i]
bmX = bmX_all[start_i:end_i]
bmY = bmY_all[start_i:end_i]

fig, ax = plt.subplots()
ax.plot(bmX, bmY, '.')
st.pyplot(fig)

phi = np.zeros([n,1])
r = np.zeros([n,1])
for i in range(0, n):
	phi[i] = math.atan2(bmY[i], bmX[i])
	r[i] = math.sqrt(bmX[i]*bmX[i] + bmY[i]*bmY[i])
      
fig, ax = plt.subplots()
ax.plot(phi, r, '.')
st.pyplot(fig)

def bernstein_base(t, i, n):
    n = n-1
    return math.comb(n,i) * math.pow(t, i) * math.pow(1 - t, n - i)

def bernstein_approx(f, t, grade = 3):
    n = f.size
    t = np.reshape(t, [n,1])
    
    A = np.zeros([n,grade])
    A_poly = np.zeros([n,grade])
    for i in range(0,grade):
        for j in range(0,n):
            A_poly[j, i] = math.pow(t[j], i)
            A[j, i] = bernstein_base(t[j], i, grade)

    ATA = A.T @ A

    b = np.reshape(f, [n,1])
    coefs = np.linalg.solve(ATA, A.T @ b)
    coefs_poly = np.linalg.solve(A_poly.T @ A_poly, A_poly.T @ b)
    return coefs, coefs_poly


with st.sidebar:
    grade = st.slider('Polynom-Grad', 1, 30, 4)
    res = st.slider('Plot Auflösung', 10, 1000, 100)

part_length = st.slider('Part Länge', 0.0, 1.0, 1/5)

indices = np.argwhere((phi + math.pi)/(2*math.pi) < part_length)

phi_part = np.take((phi + math.pi)/(2*math.pi), indices)[:,0]/part_length
r_part = np.take(r, indices)[:,0]
r_part = (r_part - np.amin(r_part))/(np.amax(r_part) - np.amin(r_part))

coefs, coefs_poly = bernstein_approx(r_part, phi_part, grade)

st.write(coefs, coefs_poly)

# fig, ax = plt.subplots()
# ax.plot(phi_part, r_part, '.')
# st.pyplot(fig)

def calc_bernsteinline(coefs, res):
    bernsteinline = np.zeros((res,2))
    grade = coefs.size

    for j in range(0, res):
        t = j/(res-1)
        b = 0
        for i in range(0, grade):
            b = b + coefs[i]*bernstein_base(t, i, grade)

        bernsteinline[j][0] = t
        bernsteinline[j][1] = b

    return bernsteinline

t_res = np.linspace(0, 1, res)

fig, ax = plt.subplots()

approx = np.zeros(res)
approx_ = calc_bernsteinline(coefs, res)
approx_poly = np.zeros(res)
st.write(approx_)
for i in range(0, grade):
    baseline = np.zeros(res)
    for j in range(0, res):
        baseline[j] = coefs[i] * bernstein_base(t_res[j], i, grade)
        approx[j] = approx[j] + baseline[j]
        approx_poly[j] = approx_poly[j] + coefs_poly[i] * math.pow(t_res[j], i)
    ax.plot(t_res, baseline)
ax.plot(phi_part, r_part, '.')
ax.plot(t_res, approx)
ax.plot(t_res, approx_poly, '--')
ax.plot(approx_[:,0], approx_[:,1], ':')
st.pyplot(fig)
     