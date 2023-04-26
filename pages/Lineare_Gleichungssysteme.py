import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Lösen von linearen Gleichungssystemen')

st.header('QR-Zerlegung')
st.write(r'''
    Sei eine Zerlegung $A = QR$ mit $Q^\top Q = \mathbb{I}$ gegeben. So kann das dazugehörige lineare Gleichungsystem $Ax = b$ durch rückwärtseinsetzen gelöst werden.
    ''')
st.latex(r'''
    Ax = QRx = b \quad \Leftrightarrow \quad Rx = Q^\top b
    ''')

st.subheader('Givens Rotation')

st.write(r'''
    Mittels Rotationen $G_i \in \mathbb{R}^{n \times n}$ wird ein Element $a_{q,p}$ der Matrix $A \in \mathbb{R}^{n \times n}$ auf null gesetzt.
    ''')
st.latex(r'''
    G_i^\top A =
	\left( \begin{array}{cccccccc}
		1 & & & & & & \\
		& \ddots & & & & 0\\
		& & c & \dots & s \\
		& & \vdots & \ddots & \vdots & & \\
		& & -s & \dots & c \\
		& 0 & & & & \ddots \\
		& & & & & & 1
	\end{array} \right)^\top
    \left( \begin{array}{cccccccc}
		* & & & \dots & & & * \\
		\vdots & \ddots & & & & & \vdots \\
		* & \dots & a_{p,p} & \dots & a_{p,q} & \dots & * \\
		\vdots & & \vdots & \ddots & \vdots & & \vdots \\
		* & \dots & a_{q,p} & \dots & a_{q,q} & \dots & *\\
		\vdots & & & & & \ddots & \vdots \\
		* & & & \dots & & & *
	\end{array} \right) \\
    = \left( \begin{array}{cccccccc}
		* & & & \dots & & & * \\
		\vdots & \ddots & & & & & \vdots \\
		\tilde{*} & \dots & \tilde{a}_{p,p} & \dots & \tilde{a}_{p,q} & \dots & \tilde{*} \\
		\vdots & & \vdots & \ddots & \vdots & & \vdots \\
		\tilde{*} & \dots & 0 & \dots & \tilde{a}_{q,q} & \dots & \tilde{*}\\
		\vdots & & & & & \ddots & \vdots \\
		* & & & \dots & & & *
	\end{array} \right)
	\qquad
	\textrm{mit}
	\begin{array}{l}
		\omega = \sqrt{ a_{p,p}^2 + a_{q,p}^2 } \\
		s = \sin(\varphi) = -\textrm{sign}(a_{p,p}) \frac{ a_{q,p} }{\omega} \\
		c = \cos(\varphi) = \frac{a_{p,p}}{\omega}
	\end{array}
    ''')
st.write(r'''
    Es gilt $G_i^\top G_i = \mathbb{I}$ und damit $G_k^\top \dots G_1^\top A = R \quad \Leftrightarrow \quad A = G_1 \dots G_k R = QR$

    Aufwand für das Anwenden einer Givens-Rotation auf eine Matrix ist in $\mathcal{O}(n)$ möglich, da sich ja jeweils nur die $q$-te und $p$-te Zeile ändert.
    ''')

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

st.subheader('Housholder Spiegelung')
st.write(r'''
    Vektor $x \in \mathbb{R}^n$ wird mit einer linearen Transformation $P \in \mathbb{R}^{n \times n}$ so gespiegelt, dass er auf einen Einheitsvektor $e \in \mathbb{R}^n$ fällt. 
    Dadurch werden alle Einträge im Vektor bis auf einen, welcher die Länge $\alpha$ enthält, gleich null. 
    __Lösung__ aufgrund unterschiedlichen Vorzeichen __nicht eindeutig__!
    ''')
st.latex(r'''
    Px = \alpha e \quad \textrm{ mit } \quad \alpha = \pm \left\Vert x \right\Vert = \pm \sqrt{ x^\top x} \\
	\textrm{Sei } P = \mathbb{I} - 2 \frac{\omega \omega^T}{\omega^\top \omega} \\
	\Rightarrow \quad Px = x - \omega \underbrace{2 \frac{\omega^\top x}{\omega^\top \omega}}_{= \lambda \in \R} = x - \lambda\omega \stackrel{!}{=} \alpha e
	\quad \Leftrightarrow \quad \omega \in \textrm{span}\left\{x - \alpha e\right\} \\
	\quad \Rightarrow \quad \omega = x \pm \sqrt{x^\top x} ~ e = \left( \begin{array}{c}
		x_1 \pm \sqrt{x^\top x} \\
		\vdots \\
		x_n
	\end{array}\right)
    ''')
st.write(r'''
    Um Auslöschungen zu vermeiden gilt:
    ''')
st.latex(r'''
    \omega = x_1 - \Vert x \Vert = \frac{\left(x_1 - \Vert x \Vert\right)\left( x_1 + \Vert x \Vert\right)}{x_1 + \Vert x \Vert} = \frac{x_1^2 - \Vert x \Vert^2}{x_1 + \Vert x \Vert} = \frac{-\left(x_2^2 + \dots + x_n^2\right)}{x_1 + \Vert x \Vert}
    ''')

def householder_transformation(A, q, p):
    n = A.shape[0]
    
    x = A[q:n,p]
    k = x.shape[0]
    x = x.reshape((k,1))
    e = np.zeros((k,1))
    e[0] = 1

    omega = x + np.sqrt(np.transpose(x) @ x) * e

    P_part = np.eye(k) - 2*(omega @ np.transpose(omega))/(np.transpose(omega) @ omega)

    P = np.eye(n)
    P[q:n,q:n] = P_part

    return P

def QR_decomposition(A):
    n = A.shape[0]

    Q = np.eye(n)
    
    for j in range(n-1):
        P = householder_transformation(A,j,j)
        A = P @ A
        Q = Q @ np.transpose(P)
        
    return Q, A

def backward_substitution(R, b):
    n = R.shape[0]
    
    x = np.zeros((n,1))
    for i in range(n-1, -1, -1):
        xi = b[i]

        for j in range(i, n, 1):
            xi = xi - R[i,j]*x[j]

        x[i] = xi/R[i,i]

    return x

def QR_solver(A, b):
    Q, R = QR_decomposition(A)
    x = backward_substitution(R, np.transpose(Q) @ b)

    return x


st.header('Iterative Verfahren')

st.subheader('Gradienten')

st.write(r'''
    Sei $A \in \mathbb{R}^{n \times n}$ symmetrisch positiv definit, 
    so kann das Gleichungssystem $Ax = b$ durch minimieren der Funktion 
    $f(x) = \frac{1}{2} x^\top A x - x^\top b$ gelöst werden.
    ''')

st.write(r'''
    Denn für ein Minimum von $f(x)$ muss $f'(x) = \nabla f(x) = Ax - b \stackrel{!}{=} 0 \Leftrightarrow Ax = b$
    gelten. Weiter gilt nach Voraussetzung $f''(x) = H_f(x) = A$ positiv definit.
    ''')

st.write(r'''
    Um nun das Minimum der Funktion $f(x)$ zu finden, gehen wir von einem Startwert $x_0 \in \mathbb{R}^n$
    in Schritten $k = 1, 2, ... $ in Richtung des steilsten Abstieges mit dem Residuums $r^{(k)}$ ($\Vert d^{(k)} \Vert = $ Fehlertoleranz) und einer Schrittlänge von $\lambda \in \mathbb{R}$.
    ''')

st.latex(r'''
    r^{(k)} = -\nabla f(x^{(k)}) = b - Ax^{(k)} \\
    \Rightarrow\quad x^{(k+1)} = x^{(k)} + \lambda r^{(k)}
    ''')

A = np.random.rand(2,2)
A = A.T @ A
# A = np.array([0.5, 0.5, 0.5, 0.9]).reshape(2,2)
A = np.array([0.5, 0.25, 0.25, 0.7]).reshape(2,2)
st.write(A)
# b = np.random.rand(2,1)
b = np.array([1, 2]).reshape(2,1)
x_sol = QR_solver(A, b)

max_it = 3
x = np.zeros((2,max_it))

grid_res = 20
grid_size = 1

# x[:,0] = np.array([
#     x_sol[0]+grid_size*(np.random.rand(1)-0.5)*2, 
#     x_sol[1]+grid_size*(np.random.rand(1)-0.5)*2
# ]).reshape(2,)
x[:,0] = np.array([1.3,1.75]).reshape(2,)

for k in range(0,max_it-1):

    x_k = x[:,k].reshape((2,1))
    r_k = b - (A @ x_k)

    Ar = A @ r_k

    step_size = (r_k.T @ r_k) / (r_k.T @ Ar)
    x_next = x_k + step_size * r_k
    x[:,k+1] = x_next.reshape(2,)

    step_res = 100
    steps = np.linspace(0, step_size*2, step_res).reshape(step_res,1)
    f_step = np.zeros(step_res)
    for i in range(0,step_res):
        x_i = x_k + steps[i] * r_k
        f_step[i] = 1/2 * x_i.T @ A @ x_i - b.T @ x_i

    fig, ax = plt.subplots()
    ax.plot(steps, f_step)
    st.pyplot(fig)

    # r_k = b - A @ x[:,k].reshape((2,1))
    
    # st.write(r_k.shape)

    # step_size = 0.1 # lambda
    # x[:,k+1] = x[:,k].reshape((2,1)) + step_size * r_k

grid_x, grid_y = np.meshgrid(
    np.linspace(x_sol[0]-grid_size, x_sol[0]+grid_size, grid_res), 
    np.linspace(x_sol[1]-grid_size, x_sol[1]+grid_size, grid_res)
)

field_u, field_v = np.meshgrid(np.zeros(grid_res), np.zeros(grid_res))

f_x = np.zeros((grid_res, grid_res))

for i in range(0,grid_res):
    for j in range(0,grid_res):
        x_ij = np.array([grid_x[i,j], grid_y[i,j]]).reshape(2,1)
        grad_ij = A @ x_ij - b
        field_u[i,j] = grad_ij[0] / np.linalg.norm(grad_ij)
        field_v[i,j] = grad_ij[1] / np.linalg.norm(grad_ij)
        f_x[i,j] = 1/2 * x_ij.T @ A @ x_ij - b.T @ x_ij


img_extent = [
    (x_sol[0]-grid_size)[0], 
    (x_sol[0]+grid_size)[0], 
    (x_sol[1]-grid_size)[0], 
    (x_sol[1]+grid_size)[0]
]

fig, ax = plt.subplots()
ax.imshow(np.flip(f_x,0), extent=img_extent)
ax.quiver(grid_x, grid_y, field_u, field_v)
ax.contour(grid_x, grid_y, f_x, 9, colors='white')
ax.plot(x[0,:], x[1,:], 'w.-')
ax.plot(x_sol[0,:], x_sol[1,:], 'o')
ax.axis('scaled')
st.pyplot(fig)




x_sol = QR_solver(A, b)
st.write(x_sol)
st.write(A @ x_sol)


st.header('Verfahren im Vergleich')
n = 5
A = np.random.rand(n,n)
st.write(A)

b = np.random.rand(n).reshape((n,1))
st.write(b)

x = QR_solver(A, b)
st.write(x)

err = np.linalg.norm(b - (A @ x))
st.write(err)

# G1 = givens_rotation(A,3,1)

# st.write(G1)
# st.write(G1 @ np.transpose(G1))
# st.write(np.transpose(G1) @ G1)
# st.write(A)

# st.write(np.transpose(G1) @ A)
