import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Wahrscheinlichkeit')

st.header('Wahrscheinlichkeitsraum')
st.write(r'Ereignisraum $\Omega$: Menge aller möglichen Ergebnisse eines Zufallsexperiments.')
st.write(r'$\sigma$-Algebra $\mathcal{A}$: Potenzmenge von $\Omega$ (im diskreten Falle).')
st.write(r'$\mathbb{P}(\omega)$: Wahrscheinlichkeitsmaß für das Ereignis $\omega \in \Omega$.')
st.write(r'$(\Omega, \mathcal{A}, \mathbb{P})$: Ist ein Wahrscheinlichkeitsraum.') 
st.write(r'''
    Sei $X$ eine Zufallsvariable, so beschreibt $\mathbb{P}(X=t) = \mathbb{P}(\{\omega \in \Omega: X(\omega) = t\})$ 
    die Wahrscheinlichkeit für die Ereignissmenge, welche alle Ereignisse enthält, wo die Zufallsvariable $X$ den Wert $t$ annimmt.
''')
st.write(r'''
    Eine Funktion $F: \mathbb{R} \rightarrow [0,1]$ mit $F_X(t) = \mathbb{P}(X \leq t)$ heißt Verteilungsfunktion
    und $f: \mathbb{R} \rightarrow [0,\infty)$ mit $F_X(t) = \int_{-\infty}^t f(x) ~dx$ heißt Dichtefunktion von $X$.
''')

st.subheader('Zufallsvektoren')
st.write(r'''
    Sei $X = (X_1, ..., X_d)$ ein Zufallsvektor, so ist die gemeinsame Verteilungsfunktion
    $F_X(t_1, ..., t_n) = \mathbb{P}(X_1 \leq t_1, ..., X_n \leq t_n)$.
    Dabei ist die **gemeinsame Verteilung des Zufallsvektors $X$ nicht durch die Verteilung der einzelnen Zufallsvariablen $X_i$ gegeben**.
    Umgekehrt kann aber die Randverteilung $F_{X_i}(t)$ (Verteilung der einzelnen Variable $X_i$) durch die gemeinsame Dichtefunktion berechnet werden.
''')
st.latex(r'''
    F_{X_i}(t) = \int_{-\infty}^t \left( \int_{-\infty}^\infty ... \int_{-\infty}^\infty f_X(x_1, ..., x_{i-1}, x_i, x_{i+1}, x_d) 
    ~dx_1 ... dx_{i-1} dx_{i+1} ~dx_d \right) ~dx_i
''')