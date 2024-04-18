import streamlit as st
import numpy as np

# A = np.array([[7/5, 1/5], [-1, 1/2]])
# st.write(A)

# eig_values = np.linalg.eigvals(A)**100
# st.write(eig_values)

# A = np.array([[-1, -1, -2], [8, -11, -8], [-10, 11, 7]])
# st.write(A)
# eig_values, eig_vector = np.linalg.eig(A)
# st.write(eig_values, eig_vector)


st.subheader('(a)')
st.write(r'''
    Sei $\tilde{x}_i$ die Projektion von $x_i$ auf $z$:
''')
st.latex(r'''
    \tilde{x}_i = \frac{\left< x_i, z \right>}{\Vert z \Vert^2} z
    \stackrel{\Vert z \Vert=1}{=} \left< x_i, z \right> z
''')

st.write(r'''
    Es soll die Distanz $\varepsilon$ zwischen $x_i$ und $\tilde{x}_i$ minimiert werden. 
    Da $z$ durch den Ursprung geht, ist durch Pythagoras bekannt, dass
''')
st.latex(r'''
    \Vert x_i \Vert^2 = \varepsilon^2 + \Vert \tilde{x}_i \Vert^2
    \quad \Leftrightarrow \quad \varepsilon^2 = \Vert x_i \Vert^2 - \Vert \tilde{x}_i \Vert^2 \\
    \Rightarrow \quad \arg \min_{\Vert z \Vert = 1} \sum_{i=1}^{10} \varepsilon^2 = 
    \arg \min_{\Vert z \Vert = 1} \sum_{i=1}^{10}\Vert x_i \Vert^2 - \Vert \tilde{x}_i \Vert^2
''')

st.write(r'''
    Da $x_i$ nicht von $z$ anhängt und für $\tilde{x}_i$ gilt
''')
st.latex(r'''
    \Vert \tilde{x}_i \Vert^2 = \Vert \left< x_i, z \right> z \Vert^2
    = \left< \left< x_i, z \right> z, \left< x_i, z \right> z \right> 
    = \left< x_i, z \right>^2 \underbrace{z^\intercal z}_{\Vert z \Vert=1} = \left< x_i, z \right>^2
''')
st.write(r'''
    folgt
''')
st.latex(r'''
    \arg \min_{\Vert z \Vert = 1} \sum_{i=1}^{10} - \Vert \tilde{x}_i \Vert^2
    \arg \min_{\Vert z \Vert = 1} \sum_{i=1}^{10} - \left< x_i, z \right>^2
    = \arg \max_{\Vert z \Vert = 1} \sum_{i=1}^{10} \left< x_i, z \right>^2
''')

st.subheader('(b)')
st.write(r'''
    Wenn $X$ zeilenweise die Datenpunkte $x_i$ enthält, so gilt
''')
st.latex(r'''
    Xz = \left( \begin{array}{cc}
        x_{1_1} & x_{1_2} \\
        \vdots & \vdots \\
        x_{10_1} & x_{10_2} \\
    \end{array} \right)\left( \begin{array}{c}
        z_1 \\
        z_2 \\
    \end{array} \right) =
    \left( \begin{array}{c}
        x_1^\intercal z \\
        \vdots \\
        x_{10}^\intercal z \\
    \end{array} \right) = 
    \left( \begin{array}{c}
        \left< x_1, z \right> \\
        \vdots \\
        \left< x_{10}, z \right> \\
    \end{array} \right) \\
    \Rightarrow \quad \Vert Xz \Vert^2 = (X z)^\intercal X z = \sum_{i=1}^{10} \left< x_i, z \right>^2
''')

st.write(r'''
    Aus der Definition der Operatornorm $\Vert X \Vert$ folgt
''')
st.latex(r'''
    \Vert X \Vert^2 = \sup_{z \in \mathbb{R}^2 \setminus \{0\}} \frac{\Vert Xz \Vert^2}{\Vert z \Vert^2}
    = \sup_{\Vert z \Vert = 1} \Vert Xz \Vert^2
    = \max_{\Vert z \Vert = 1} \sum_{i=1}^{10} \left< x_i, z \right>^2
''')
st.write(r'''
    Sei $\sigma$ der größte Singulärwert von $X$, so gilt nach Definition $\Vert X \Vert = \sigma \Rightarrow \Vert X \Vert^2 = \sigma^2$.
''')

st.subheader('(c)')
st.write(r'''
    Sei $X = U \Sigma V^\intercal$ die Singulärwertzerlegung von $X$ und $v_1$ die erste Spalte von $V$.
    Da $V$ eine orthonormale Matrix ist, gilt für $v_1^\intercal v_1 = 1$ und $v_1^\intercal v_i = 0$ für $i \neq 1$ und damit
''')
st.latex(r'''
    \Vert X v_1 \Vert^2
    = \Vert U \Sigma V^\intercal v_1 \Vert^2
    = \left\Vert U
    \left(\begin{array}{cccc}
        \sigma_1 \\
        & \sigma_2 \\
        & & \ddots \\
        & & & \sigma_{10} \\
    \end{array}\right)
    \left(\begin{array}{c}
        v_1^\intercal v_1 \\
        v_2^\intercal v_1 \\
        \vdots \\
        v_{10}^\intercal v_1 \\
    \end{array}\right) \right\Vert^2 \\
    = \left\Vert 
    \left(\begin{array}{ccc}
        | & & | \\
        u_1 & ... & u_{10} \\
        | & & | \\
    \end{array}\right)
    \left(\begin{array}{c}
        \sigma_1 \\
        \vdots \\
        0 \\
    \end{array}\right) \right\Vert^2
    = \Vert u_1 \sigma_1 \Vert^2
    = \underbrace{\Vert u_1 \Vert^2}_{=1} \sigma_1^2 = \sigma_1^2
''')

