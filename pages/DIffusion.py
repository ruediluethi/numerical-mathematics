import streamlit as st
import numpy as np

from pages.Lineare_Gleichungssysteme import QR_solver

n = int(10)

A = np.eye(n*n, dtype=float)*4

F = np.zeros((n,n), dtype=float)
F[int(n/2),int(n/2)] = 10.0

st.write(F)

b = np.zeros((n*n,1), dtype=float)
for i in range(0,n):
    for j in range(0,n):

        k = i*n + j
        b[k] = F[i,j]
        
        wi = (k-1) % (n*n)
        ei = (k+1) % (n*n)
        nj = (k-n) % (n*n)
        sj = (k+n) % (n*n)
        
        A[k,wi] = -1
        A[k,ei] = -1
        
        A[k,nj] = -1
        A[k,sj] = -1
            

# x = QR_solver(A, b)

x = np.linalg.solve(A, b)

st.write(x)

for i in range(0,n):
    for j in range(0,n):

        k = i*n + j
        F[i,j] = x[k]

st.write(F)