import streamlit as st
import numpy as np

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
