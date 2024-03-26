import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Eigenwerte, Eigenvektoren und Diagonalisierbarkeit')

st.header('Eigenwerte und Eigenvektoren')

st.subheader('Definition')
st.write(r'''
    Sei $A \in \mathbb{K}^{n \times n}$. Gibt es eine Zahl $\lambda \in \mathbb{K}$
    und einen Vektor $v \in \mathbb{K}^n, v \neq 0$ mit der Eigenschaft
''')
st.latex(r'''
    Av = \lambda v  
''')
st.write(r'''
    so heißt $\lambda$ **Eigenwert** zum **Eigenvektor** $v$.
    Die Menge $\sigma(A)$ aller Eigenwerte einer Matrix $A$ heißt **Spektrum** von $A$.
''')

st.write(r'''
    Die Menge aller Eigenvektoren $E(\lambda)$ zu einem Eigenwert $\lambda$ heißt **Eigenraum** und 
    ist ein [Untervektorraum](https://de.wikipedia.org/wiki/Untervektorraum) von $\mathbb{K}^n$, denn
''')
st.latex(r'''
    E(\lambda) \neq \varnothing \\
    v, w \in E(\lambda), A(v+w) = \lambda v + \lambda w = \lambda (v + w) \Rightarrow v+w \in E(\lambda) \\
    \alpha \in \mathbb{K}, A \left( \alpha \cdot v \right) = \alpha Av = \alpha \lambda v = \lambda \left(\alpha v \right) \Rightarrow v+w \in E(\lambda)
''')
st.write(r'''
    Die Dimension von $E(\lambda)$ heißt **geometrische Vielfachheit** des Eigenvektors $\lambda$.      
''')
st.write(r'''
    **Satz** Seien $v_1, ..., v_k \in \mathbb{K}$ Eigenvektoren von $A \in \mathbb{K}^{n \times n}$ mit Eigenwerten $\lambda_1, ..., \lambda_k \in \mathbb{K}$.
    Sind $\lambda_1, ..., \lambda_k$ paarweise verschieden, so sind $v_1, ..., v_k$ linear unabhängig.
''')
st.write(r'''
    Sei $A \in \mathbb{K}^{n \times n}$ und $I \in \mathbb{K}^{n \times n}$ die Einheitsmatrix, 
    dann ist $\lambda$ genau dann ein Eigenwert des **charakteristische Polynom** wenn es sich dabei um eine Nullstelle handelt:
''')
st.latex(r'''
    p_A(\lambda) = \det(A - \lambda I) = 0
''')
