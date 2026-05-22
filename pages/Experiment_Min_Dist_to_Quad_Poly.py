import streamlit as st
import numpy as np
import math 
from matplotlib import pyplot as plt


st.write(r'''
    Sei $p \in \mathbb{R}^2$ ein Punkte in der Ebene und sei eine quadratische Kurve definiert durch      
''')
st.latex(r'''
    f(t) = \left( \begin{array}{c}
        a_0 + a_1 t + a_2 t^2 \\
        b_0 + b_1 t + b_2 t^2
    \end{array} \right)
''')

st.write(r'''
    Der Vektor vom Punkt $p$ und zum auf der Kurve am nächsten liegenden Punkt $\tilde{f} = f(\tilde{t})$ 
    muss orthogonal zur Tangente $f'(t)$ der Kurve zum Zeitpunkt $\tilde{t}$ sein.
    Also muss das Skalarprodukt der beiden Vektoren $0$ betragen:
''')
st.latex(r'''
   \left( f(\tilde{t}) - p \right)^\top f'(\tilde{t}) \stackrel{!}{=} 0
''')
st.write(r'''
    Ausmultipliziert
''')


# p = np.array([[1.2, 2.3], [4.2, 1.3], [3.2, 2.6]])
s = np.random.rand(3, 2)
p = np.random.rand(1, 2)

fig, ax = plt.subplots()

ax.plot(s[:,0], s[:,1], '.')
ax.plot(p[0,0], p[0,1], 'o')

grade = s.shape[0]

def create_vandermonde(grade, t):
    A = np.zeros((t.size, grade))
    for i in range(t.size):
        for j in range(grade):
            A[i,j] = np.pow(t[i], j)
    return A

A = create_vandermonde(grade, np.linspace(0.0, 1.0, grade))
# st.write(A)

a = np.linalg.solve(A, s[:,0])
b = np.linalg.solve(A, s[:,1])
# st.write(a, b)

st.latex(r'''
    \left( \begin{array}{c}
        a_0 + a_1 \tilde{t} + a_2 \tilde{t}^2 - p_x \\
        b_0 + b_1 \tilde{t} + b_2 \tilde{t}^2 - p_y
    \end{array} \right)^\top
    \left( \begin{array}{c}
        a_1 + 2 a_2 \tilde{t} \\
        b_1 t + 2 b_2 \tilde{t}
    \end{array} \right) \\
    = \left( a_0 + a_1 \tilde{t} + a_2 \tilde{t}^2 - p_x \right) \left( a_1 + 2 a_2 \tilde{t} \right) +
    \left( b_0 + b_1 \tilde{t} + b_2 \tilde{t}^2 - p_y \right) \left( b_1 t + 2 b_2 \tilde{t} \right) \\
    = a_0 a_1 + 2 a_0 a_2 \tilde{t}
    + a_1^2 \tilde{t} + 2 a_1 a_2 \tilde{t}^2
    + a_1 a_2 \tilde{t}^2 + 2 a_2^2 \tilde{t}^3
    - p_x a_1 - 2 p_x a_2 \tilde{t} \\
    + b_0 b_1 + 2 b_0 b_2 \tilde{t}
    + b_1^2 \tilde{t} + 2 b_1 b_2 \tilde{t}^2
    + b_1 b_2 \tilde{t}^2 + 2 b_2^2 \tilde{t}^3
    - p_y b_1 - 2 p_y b_2 \tilde{t} \\
    = 2 \left( a_2^2 + b_2^2 \right) \tilde{t}^3 \\
    + 3 \left( a_1 a_2 + b_1 b_2 \right) \tilde{t}^2 \\
    + \left( \left(a_1^2 + b_1^2 \right) + 2 \left( a_2 \left(a_0 - p_x \right) + b_2 \left(b_0 - p_y \right) \right) \right) \tilde{t} \\
    + a_1 \left(a_0 - p_x\right) + b_1 \left( b_0 - p_y \right)
''')

roots = np.roots([2*(a[2]**2 + b[2]**2),
                  3*(a[1]*a[2] + b[1]*b[2]),
                  (a[1]**2 + b[1]**2) + 2*(a[2]*(a[0] - p[0,0]) + b[2]*(b[0] - p[0,1])),
                  a[1]*(a[0] - p[0,0]) + b[1]*(b[0] - p[0,1])])

st.write(roots)

real_roots = np.array([])
closest_points = np.zeros((0,2))
for root in roots:
    if np.isreal(root):
        t = root.real
        real_roots = np.append(real_roots, t)
        # if t >= 0 and t <= 1:
        closest_points = np.vstack([closest_points, np.array([[a[0] + a[1]*t + a[2]*t**2, b[0] + b[1]*t + b[2]*t**2]])])
        ax.plot(closest_points[-1,0], closest_points[-1,1], 'x')
        ax.plot([p[0,0], closest_points[-1,0]], [p[0,1], closest_points[-1,1]], 'k--')

res = 100

A = create_vandermonde(grade, np.linspace(np.amin(real_roots), np.amax(real_roots), res))
f_x = A @ a
f_y = A @ b
ax.plot(f_x, f_y, ':')

A = create_vandermonde(grade, np.linspace(0.0, 1.0, res))
f_x = A @ a
f_y = A @ b
ax.plot(f_x, f_y)
ax.set_aspect('equal')

st.pyplot(fig)