import os
import streamlit as st
import numpy as np
import pandas as pd

from pages.Lineare_Gleichungssysteme import *

st.title('Interpolation und Approximation')


path_to_data = os.path.join('data', 'aggregated_data_export_20230315T112419880')

csv_list = os.listdir(path_to_data)

data = np.genfromtxt(os.path.join(path_to_data, csv_list[0]), delimiter=',')

t = data[:,0]
bm = data[:,21]

K = st.slider('Grad des Polynoms', 1, 100, 30)
n = bm.size
bm = np.reshape(bm, [n,1])

t_max = np.amax(t)

# fill in Vandermonde matrix
V_poly = np.zeros([n,K])
for i in range(0,n):
  for j in range(0,K):
    V_poly[i][j] = np.power(t[i]/t_max,j)

# method of least square
A = V_poly.T @ V_poly
b = V_poly.T @ bm

# solve
coefs = np.linalg.solve(A, b)
# coefs = QR_solver(A, b)

def eval_poly(t, coefs):
  res = 0
  for i in range(0,len(coefs)):
    res = res + coefs[i] * np.power(t,i)
  return res

alpha = 0.98
delta_t = 0

bm_interp = np.zeros([n,1])
for i in range(0,n):
  t_i = i/n
  bm_interp[i] = eval_poly(max(0, min(1, alpha * (t_i + delta_t))), coefs)

st.line_chart(pd.DataFrame(np.array([bm[:,0], bm_interp[:,0]]).T))

# st.line_chart(pd.DataFrame([bm, bm_interp]))