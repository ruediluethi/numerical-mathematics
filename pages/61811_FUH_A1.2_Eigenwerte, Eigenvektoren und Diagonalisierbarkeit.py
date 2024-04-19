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
    p_A(\lambda) = \det(A - \lambda I) \stackrel{!}{=} 0
''')
st.write(r'''
    Sei $S_{ij}(A)$ die Streichungsmatrik von $A \in \mathbb{K}^{n \times n}$ ohne die $i$-te Zeile und $j$-te Spalte:
''')
st.latex(r'''
    S_{ij}(A) =
	\left( \begin{array}{ccccccc}
         &  & a_{1(j-1)} & a_{1(j+1)} &  &  \\
         &  & \vdots & \vdots &  &  \\
		a_{(i-1)1} & \cdots & a_{(i-1)(j-1)} & a_{(i-1)(j+1)} & \cdots & a_{(i-1)n} \\
        a_{(i+1)1} & \cdots & a_{(i+1)(j-1)} & a_{(i+1)(j+1)} & \cdots & a_{(i+1)n} \\
         &  & \vdots & \vdots &  &  \\
         &  & a_{n(j-1)} & a_{n(j+1)} &  &  \\
	\end{array} \right)     
''')
st.write(r'''
    Damit gilt für $p_A(\lambda)$ durch eine Entwicklung der Determinante nach erster Spalte
''')
st.latex(r'''
    \det(A - \lambda I) = \det(B) = \sum_{k=1}^{n} (-1)^{1+k} b_{k1}  \det\left(S_{k1}(B)\right) \\
    \stackrel{b_{kk} = a_{kk} - \lambda}{=} \left( a_{11} - \lambda \right) \det\left(S_{k1}(B)\right) + \sum_{k=2}^{n} (-1)^{1+k} a_{k1}  \det\left(S_{k1}(B))\right) \\
    = \left( a_{11} - \lambda \right) \sum_{k=1}^{n-1} (-1)^{1+k} b_{(k+1)2} \det\left(S_{k1}(S_{k1}(B))\right) + ... \\
    \stackrel{b_{kk} = a_{kk} - \lambda}{=} \left( a_{11} - \lambda \right) \left( a_{22} - \lambda \right) \det\left(S_{k1}(S_{k1}(B))\right) + ... \\
    = ... = \left( a_{11} - \lambda \right) \left( a_{22} - \lambda \right) ... \left( a_{nn} - \lambda \right) + ... = p_A(\lambda)
''')
st.write(r'''
    wodurch klar wird, dass $p_A(\lambda)$ ein Polynom $n$-ten Grades ist.
''')
st.write(r'''
    *Beweis* Ist $p_A(\lambda) = \det(A - \lambda I) = 0$ so ist $A - \lambda I$ singulär und es existieren lineare abhängige Spalten/Zeilen-Vektoren.
    Nur dann kann $\exists v \neq 0$ mit $\left(A - \lambda I\right)v = Av - \lambda I v = 0 \Leftrightarrow Av = \lambda v$ gelten.
''')
st.write(r'''
    **Fundamentalsatz der Algebra** Jedes Polynom $P(z): \mathbb{C} \rightarrow \mathbb{C}$ vom Grad $n$ hat genau $n$ Nullstellen $\lambda_1, ..., \lambda_n \in \mathbb{C}$
    und kann folgendermaßen dargestellt werden $P(z) = (z - \lambda_1) ... (z - \lambda_n)$
''')
st.write(r'''
    Seien $\lambda_1, ..., \lambda_k$ paarweise verschiedene Nullstellen von $P(z) = (z - \lambda_1)^{i_1}...(z - \lambda_k)^{i_k}$
    so heißt $i_j$ die **algebraische Vielfachheit** des Eigenwertes $\lambda_j$ und 
    für eine Matrix $A \in \mathbb{C}^{n \times n}$ gilt $\sum_{j=1}^{k} i_j = n$.
''')
st.write(r'''
    **Satz** Die Eigenwerte $\lambda$ einer hermiteschen Matrix $A \in \mathbb{C}^{n \times n}$ mit $\overline{A}^\intercal = A^* = A$ sind reell.
''')
st.write(r'''
   *Beweis* Für das komplexe [Skalarprodukt](https://de.wikipedia.org/wiki/Standardskalarprodukt#Skalarprodukt-Axiome_2) gilt 
    $\left< \lambda v, w \right> = \overline{\lambda}$ und $\left< v, \lambda w \right> = \lambda$
    damit gilt:
''')
st.latex(r'''
    \lambda \left< v, v \right> = \left< v, \lambda v \right> = \left< v, A v \right>
    = v^* \left(A v\right) \stackrel{A = A^*}{=} v^* A^* v \\
    = \left(A v\right)^* v = \left< A v, v \right> = \left< \lambda v, v \right> = \overline{\lambda} \left< v, v \right>\\
    \Rightarrow \quad \lambda = \overline{\lambda} \quad
    \Rightarrow \quad \lambda \in \mathbb{R}
''')
st.write(r'''
   *Korollar* Die Eigenwerte $\lambda$ einer symmetrischen Matrix $A \in \mathbb{R}^{n \times n}$ mit $A^\intercal = A$ sind reell.
''')
st.write(r'''
   Sei $v = a + ib \in \mathbb{C}^n$ ein komplexer Eigenvektor 
''')
