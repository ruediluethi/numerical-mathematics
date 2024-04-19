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
    Sei $A \in \mathbb{R}^{n \times n}$ symmetrisch positiv definit.
    So kann das Gleichungssystem $Ax = b$ durch minimieren der Funktion 
    $f(x) = \frac{1}{2} x^\top A x - x^\top b$ gelöst werden.
    ''')

st.write(r'''
    Denn für ein Minimum von $f(x)$ muss die Ableitung/Gradient
    $f'(x) = \nabla f(x) = Ax - b \stackrel{!}{=} 0 \Leftrightarrow Ax = b$
    gleich Null sein.
    Weiter gilt nach Voraussetzung $f''(x) = H_f(x) = A$ positiv definit.
    ''')

st.write(r'''
    Um nun das Minimum der Funktion $f(x)$ zu finden, 
    gehen wir von einem Startwert $x_0 \in \mathbb{R}^n$
    in Richtung des steilsten Abstieges $r_0 = -\nabla f(x_0) = b - Ax_0$.
    ''')

grid_res = 20
grid_size = 1

A_default = np.array([0.9, 0.65, 0.65, 1]).reshape(2,2)
b_default = np.array([1, 2]).reshape(2,1)
# A_default = np.array([2, 1, -1, 2]).reshape(2,2)
# st.write(A_default)
# st.write(A_default.T)
# b_default = np.array([3, 4]).reshape(2,1)

x_0_default = np.array([-1.35,2.95])

# init session states with default values
if 'A' not in st.session_state:
    st.session_state.A = A_default
if 'b' not in st.session_state:
    st.session_state.b = b_default
if 'x_0' not in st.session_state:
    st.session_state.x_0 = x_0_default

if (st.button('Beispielwerte wiederherstellen')):
    st.session_state.A = A_default
    st.session_state.b = b_default
    st.session_state.x_0 = x_0_default

if (st.button('Zufälliges Gleichungssystem Ax = b generieren')):
    A = np.random.rand(2,2)
    A = A.T @ A
    st.session_state.A = A
    b = np.random.rand(2,1)
    st.session_state.b = b
    x_sol = QR_solver(A, b)
    st.session_state.x_0 = np.array([
        x_sol[0]+grid_size*(np.random.rand(1)-0.5)*2, 
        x_sol[1]+grid_size*(np.random.rand(1)-0.5)*2
    ])

A = st.session_state.A
b = st.session_state.b

if (st.button('Zufälliger Startwert generieren')):
    x_sol = QR_solver(A, b)
    st.session_state.x_0 = np.array([
        x_sol[0]+grid_size*(np.random.rand(1)-0.5)*2, 
        x_sol[1]+grid_size*(np.random.rand(1)-0.5)*2
    ])

# generate grid to plot vectorfield
x_sol = QR_solver(A, b)

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

# calc first step
x_0 = st.session_state.x_0.reshape(2,1)
r_0 = b - A @ x_0

lambda_min = (r_0.T @ r_0) / (r_0.T @ A @ r_0)

x_1 = x_0 + lambda_min * r_0
x_end = x_0 + lambda_min * 1.5 * r_0

fig, ax = plt.subplots()
ax.set_title(r'f(x)')
ax.imshow(np.flip(f_x,0), extent=img_extent, cmap='Greys')
ax.quiver(grid_x, grid_y, field_u, field_v, color='grey', label=r'Gradient $\nabla f(x)$')
ax.contour(grid_x, grid_y, f_x, 9, colors='black')
ax.plot(x_sol[0,:], x_sol[1,:], 'kx', label=r'$A^{-1}b$')
ax.plot([x_0[0,:], x_end[0,:]], [x_0[1,:], x_end[1,:]], 'k-', label=r'$F(\lambda) = x_0 + \lambda r_0$')
ax.plot(x_0[0,:], x_0[1,:], 'ko', label=r'$x_0$')
ax.plot(x_1[0,:], x_1[1,:], 'k.', label=r'$x_1$')
ax.axis('scaled')
ax.legend()
st.pyplot(fig)


st.write(r'''
    Entlang der Geraden $x_0 + \lambda r_0$ wird der Funktionswert
    $F(\lambda) = f(x_0 + \lambda r_0)$ für ein $\lambda \in \mathbb{R}$
    erstmals kleiner. Wobei das Minimum von $f(x)$ nicht erreicht wird.
    Also wird am Minimum von $F(\lambda_{\min}) \rightarrow \min$ 
    für $x_1 = x_0 + \lambda_{\min} r_0$ erneut die Abstiegsrichtung
    bestimmt $r_1 = -\nabla f(x_1) = b - Ax_1$.
    ''')

step_res = 100
steps = np.linspace(0, lambda_min[0] * 1.5, step_res)
F_of_lambda = np.zeros(step_res)
for i in range(0,step_res):
    x_lambda = x_0 + steps[i] * r_0
    F_of_lambda[i] = 1/2 * x_lambda.T @ A @ x_lambda - b.T @ x_lambda

x_lambda_min = x_0 + lambda_min * r_0
F_of_lambda_min = 1/2 * x_lambda_min.T @ A @ x_lambda_min - b.T @ x_lambda_min

fig, ax = plt.subplots()
ax.plot(steps, F_of_lambda, 'k', label=r'$F(\lambda)$')
ax.plot(lambda_min, F_of_lambda_min, 'ko', label=r'$\lambda_{\min}$')
ax.legend()
st.pyplot(fig)

st.write(r'''
    Die Funktion $F: \lambda \in \mathbb{R} \rightarrow \mathbb{R}$ ist 
    eine quadratische reellwertige Funktion.
    ''')
st.latex(r'''
    F(\lambda) = f(x + \lambda r) = \frac{1}{2} \left( x + \lambda r \right)^\top A \left( x + \lambda r \right) - \left( x + \lambda r \right)^\top b \\
	= \frac{1}{2} \left( x^\top A x + \underbrace{x^\top A \lambda r + \lambda r^\top A x}_{= ~ 2\lambda ~ r^\top A x \textrm{ (da skalar)}} + \lambda r^\top A \lambda r \right) - \left( x^\top b + \lambda r^\top b \right) \\
	= \frac{1}{2} \lambda^2 r^\top A r + \lambda\left( r^\top A x- r^\top b \right) + \underbrace{\frac{1}{2}x^\top A x - x^\top b}_{=~f(x)}
    ''')
st.write(r'''
    Weiter gilt $F(\lambda_{\min}) \rightarrow \min \Leftrightarrow F'(\lambda_{\min}) \stackrel{!}{=} 0$.
    ''')
st.latex(r'''
    F'(\lambda) = \lambda r^\top A r  + r^\top A x - r^\top b \stackrel{!}{=} 0 \\
	\Rightarrow \lambda_{\min} = \frac{r^\top b - r^\top A x}{r^\top A r} 
	= \frac{r^\top \overbrace{(b - Ax)}^{=~r}}{r^\top A r}
	= \frac{\left\langle r, r \right\rangle}{\left\langle r, Ar \right\rangle}
    ''')

st.write(r'''
    Das Verfahren wird solange wiederholt, bis für den Betrag das Residuum $r = b - Ax$ 
    eine gewisse Toleranz $\left\Vert r \right\Vert < \textrm{tol}$ unterschritten ist.
    ''')

max_it = 30

x_sol = QR_solver(A, b)
x = np.zeros((2,max_it))
x[:,0] = x_0.reshape(2,)


for k in range(0,max_it-1):

    x_k = x[:,k].reshape((2,1))
    r_k = b - (A @ x_k)

    Ar = A @ r_k

    step_size = (r_k.T @ r_k) / (r_k.T @ Ar)
    x_next = x_k + step_size * r_k
    x[:,k+1] = x_next.reshape(2,)

fig, ax = plt.subplots()
ax.set_title(r'f(x)')
ax.imshow(np.flip(f_x,0), extent=img_extent, cmap='Greys')
ax.quiver(grid_x, grid_y, field_u, field_v, color='grey')
ax.contour(grid_x, grid_y, f_x, 9, colors='black')
ax.plot(x_sol[0,:], x_sol[1,:], 'kx', label=r'$A^{-1}b$')
ax.plot(x[0,:], x[1,:], 'k.-', label=r'$x_0, x_1, ...$')
ax.axis('scaled')
ax.legend()
st.pyplot(fig)

st.subheader('Konjugierte Gradienten')

st.write(r'''
    Das Energieskalarprodukt $\left\langle x,y \right\rangle_A = x^\top A y = 
    \left\langle x, Ay \right\rangle_2$ wird verwendet um mittels Gram-Schmidt-Verfahren
    eine Orthogonale Basis für $Ax = b$ zu erzeugen. 
    ''')

#cg method

# max_it = 3

x = np.zeros((2,max_it))
x[:,0] = x_0.reshape(2,)

r_k = b - A @ x_0
rho = r_k.T @ r_k
d_k = r_k

for k in range(1,max_it):
    
    a_k = A @ d_k
    alpha = rho / (d_k.T @ a_k)

    x_prev = x[:,k-1].reshape((2,1))
    x_k = x_prev + alpha * d_k
    x[:,k] = x_k.reshape(2,)

    r_k = r_k - alpha * a_k
    rho_next = r_k.T @ r_k
    d_k = r_k + rho_next/rho * d_k
    rho = rho_next

fig, ax = plt.subplots()
ax.set_title(r'f(x)')
ax.imshow(np.flip(f_x,0), extent=img_extent, cmap='Greys')
ax.quiver(grid_x, grid_y, field_u, field_v, color='grey')
ax.contour(grid_x, grid_y, f_x, 9, colors='black')
ax.plot(x_sol[0,:], x_sol[1,:], 'kx', label=r'$A^{-1}b$')
ax.plot(x[0,:], x[1,:], 'k.-', label=r'$x_0, x_1, x_2 = A^{-1}b$')
ax.axis('scaled')
ax.legend()
st.pyplot(fig)



# st.header('Verfahren im Vergleich')
n = 5
A = np.random.rand(n,n)
# st.write(A)

b = np.random.rand(n).reshape((n,1))
# st.write(b)

x = QR_solver(A, b)
# st.write(x)

err = np.linalg.norm(b - (A @ x))
# st.write(err)

# G1 = givens_rotation(A,3,1)

# st.write(G1)
# st.write(G1 @ np.transpose(G1))
# st.write(np.transpose(G1) @ G1)
# st.write(A)

# st.write(np.transpose(G1) @ A)
