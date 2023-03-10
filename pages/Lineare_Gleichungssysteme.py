import streamlit as st
import pandas as pd
import numpy as np

n = 5

B = np.eye(n)
B[3,1] = 2

st.write(B)

A = np.random.rand(n,n);

def givens_rotation(A, q, p):
    n = A.shape[0]

    a_qp = A[q,p]
    a_pp = A[p,p]

    omega = np.sqrt( a_pp**2 + a_qp**2 )
    s = -np.sign(a_pp)*a_qp/omega
    c = a_pp/omega

    G = np.eye(n)
    G[p,p] = c
    G[p,q] = s
    G[q,p] = -s
    G[q,q] = c

    return G


G1 = givens_rotation(A,3,1)

st.write(G1)
st.write(A)

st.write(np.transpose(G1) @ A)
