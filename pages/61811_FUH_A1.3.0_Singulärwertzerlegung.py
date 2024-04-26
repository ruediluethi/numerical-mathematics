import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# A = np.array([[7/5, 1/5], [-1, 1/2]])
# st.write(A)

# eig_values = np.linalg.eigvals(A)**100
# st.write(eig_values)

# A = np.array([[-1, -1, -2], [8, -11, -8], [-10, 11, 7]])
# st.write(A)
# eig_values, eig_vector = np.linalg.eig(A)
# st.write(eig_values, eig_vector)

st.title('Singulärwertzerlegung')

st.header('Anwendung: Matrixapproximation')

X = np.array([[-2.18, 1.92, 5.72, -11.28, 4.62, -0.38, 2.82, -4.78, 3.42,  0.12],
              [-1.19, 3.31, 2.41,  -4.49, 2.61,  0.51, 5.91, -4.79, 0.91, -5.19]])


st.write(r'''
    Sei $X$ eine Matrix, welche die Datenpunkte $x_i$ in den Spalten enthält:
''')
st.write(X)



st.subheader('Optimierungsproblem')
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

st.subheader('Matrixnorm')
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

st.subheader('Lösung durch Singulärwertzerlegung')
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

st.header('Singulärwertzerlegung')

st.write(r'''
    Gesucht ist eine Zerlegung der Matrix $A \in \mathbb{R}^{m \times n}$ in die Matrizen 
    $U \in \mathbb{R}^{m \times m}, \Sigma \in \mathbb{R}^{m \times n}, V \in \mathbb{R}^{n \times n}$
    sodass $A = U \Sigma V^\intercal$ gilt.
''')

# A = np.array([[5, 3], [4, 1], [1, 4]])
A = X.T
m = A.shape[0]
n = A.shape[1]

ATA = A.T @ A
# st.write(ATA)

st.subheader('Berechnen der Eigenwerte')
st.write(r'''
    Der symmetrisch positiv definiten Matrix $A^\intercal A$.
''')
st.latex(r'''
    P(\lambda) = \det \left( A^\intercal A - \lambda I \right)
    = \det \left( \begin{array}{cc}
        a_{11} - \lambda & a_{12} \\
        a_{21} & a_{22} - \lambda \\
    \end{array} \right) \\
    = \left( a_{11} - \lambda \right)\left( a_{22} - \lambda \right) - a_{12} a_{21}
    = \lambda^2 - \lambda \left( a_{11} + a_{22} \right) - a_{12} a_{21} + a_{11} a_{22} \\
    \Rightarrow \quad \lambda_{1,2} = \frac{\left( a_{11} + a_{22} \right) \pm \sqrt{\left( a_{11} + a_{22} \right)^2 + 4 a_{12} a_{21} - 4a_{11} a_{22} }}{2}
''')

a = 1
b = -(ATA[0,0] + ATA[1,1])
c = - ATA[0,1] * ATA[1,0] + ATA[0,0] * ATA[1,1]
lambda_1 = ((ATA[0,0] + ATA[1,1]) + np.sqrt((ATA[0,0] + ATA[1,1])**2 + 4 * ATA[0,1]*ATA[1,0] - 4*ATA[0,0]*ATA[1,1]) )/2
lambda_2 = ((ATA[0,0] + ATA[1,1]) - np.sqrt((ATA[0,0] + ATA[1,1])**2 + 4 * ATA[0,1]*ATA[1,0] - 4*ATA[0,0]*ATA[1,1]) )/2

st.write(r'$\lambda_1 = $'+str(round(lambda_1, 3)))
st.write(r'$\lambda_2 = $'+str(round(lambda_2, 3)))

# st.write(lambda_1, lambda_2)

st.subheader('Berechnen der Eigenvektoren')
st.latex(r'''
    \left( A^\intercal A - \lambda I \right) v \stackrel{!}{=} 0 \\
    \Rightarrow \quad \left( \begin{array}{cccc}
        a_{11} - \lambda & a_{12} & | & 0 \\
        a_{21} & a_{22} - \lambda & | & 0
    \end{array} \right) \\
    \Rightarrow \quad \left( \begin{array}{cccc}
        a_{11} - \lambda & a_{12} & | & 0 \\
        a_{21} - \frac{a_{21}}{a_{11} - \lambda} (a_{11} - \lambda) & 
        a_{22} - \lambda - \frac{a_{21}}{a_{11} - \lambda} a_{12} & | & 0
    \end{array} \right) \\
''')

A_1 = ATA - np.eye(2) * lambda_1
A_1[1,:] = A_1[1,:] - A_1[1,0]/A_1[0,0] * A_1[0,:]
v_1 = np.array([[-A_1[0,1]/A_1[0,0]],[1]])
v_1 = v_1/np.linalg.norm(v_1)

A_2 = ATA - np.eye(2) * lambda_2
A_2[1,:] = A_2[1,:] - A_2[1,0]/A_2[0,0] * A_2[0,:]
v_2 = np.array([[1],[-A_2[0,0]/A_2[0,1]]])
v_2 = v_2/np.linalg.norm(v_2)

V = np.zeros((2,2))
V[:,0] = v_1.flatten()
V[:,1] = v_2.flatten()

D = np.zeros((2,2))
D[0,0] = lambda_1
D[1,1] = lambda_2

# st.write(D, V)
# st.write(V.T @ D @ V)

Sigma = np.zeros((m,n))
Sigma[0,0] = np.sqrt(lambda_1)
Sigma[1,1] = np.sqrt(lambda_2)

# U, S, Vh = np.linalg.svd(A)
# st.write(U, S, Vh)

st.subheader(r'Berechnen der Matrix $U$')


st.write(r'''
    Für die Spalten $u_i$ mit $i = 1, ..., n$ von $U$ gilt
''')
st.latex(r'''
    u_i = \frac{A v_i}{\Vert A v_i \Vert} 
    = \frac{A v_i}{\sqrt{ v_i^\intercal A^\intercal A v_i }}
    = \frac{A v_i}{\sqrt{ v_i^\intercal \lambda v_i }} 
    = \frac{A v_i}{\sqrt{ \lambda } \underbrace{\Vert v_i \Vert}_{=1}} 
    = \frac{1}{\sqrt{\lambda_i}} A v_i
''')

st.write(r'''
    Die Spalten $u_k$ mit $k = {n+1}, ..., m$ werden mittels der Einheitsvektoren $e_{n+1}, ..., e_m$ und dem Gramm-Schmidt-Verfahren ergänzt.
    Sei $\tilde{u}_k$ eine Linarkombination aus den bisherigen Spalten $u_1, ..., u_{k-1}$ und $e_k$
''')
st.latex(r'''
    \tilde{u}_k = e_k + \sum_{j=1}^{k-1} \alpha_j u_j
''')
st.write(r'''
    Wobei die neue Spalte $\tilde{u}_k$ orthogonal zu den bisherigen Spalten $u_1, ..., u_{k-1}$ sein muss.
''')
st.latex(r'''
    \forall i \in \{1, ..., k-1\}: \quad \left< u_i, \tilde{u}_k \right> 
    = \left< u_i, e_k + \sum_{j=1}^{k-1} \alpha_j u_j \right>
    = \left< u_i, e_k \right> + \underbrace{\sum_{j=1}^{k-1} \alpha_j \left< u_i, u_j \right>}_{\substack{i=j \Rightarrow 1 \\ i \neq j \Rightarrow 0}} = 0 \\
    \Leftrightarrow \quad \alpha_i = - \left< u_i, e_k \right>
''')



U = np.zeros((m,m))
for i in range(0,n):
    U[:,i] = 1/np.sqrt(D[i,i]) * A @ V[:,i]

# st.write(U)

for k in range(n,m):
    e_k = np.zeros(m)
    e_k[k] = 1
    u_k_ = e_k
    for j in range(0,k):
        u_k_ = u_k_ - U[:,j].T @ e_k * U[:,j]
      
    U[:,k] = u_k_/np.linalg.norm(u_k_)

# st.write(U)


# st.write(Sigma)

# st.write(X)
# st.write((U @ Sigma @ V.T).T)

# st.write(X)

# st.write(np.sqrt(X[0,0]**2 + X[1,0]**2))

max_scale = np.amax((np.sqrt(X[0,:]**2 + X[1,:]**2)))

normed_Sigma = Sigma / np.amax(Sigma)

res = 100
phi = np.linspace(0, 2*np.pi, res)
circle = np.zeros((2, res))
circle[0,:] = np.cos(phi)
circle[1,:] = np.sin(phi)
oval = (V @ normed_Sigma[0:2,0:2] @ V.T) @ circle * max_scale

fig, ax = plt.subplots()
ax.plot(X[0,:], X[1,:], 'k.')
ax.plot(np.array([-V[0,0], V[0,0]])*normed_Sigma[0,0]*max_scale, 
        np.array([-V[1,0], V[1,0]])*normed_Sigma[0,0]*max_scale, 'k')
ax.plot(np.array([-V[0,1], V[0,1]])*normed_Sigma[1,1]*max_scale, 
        np.array([-V[1,1], V[1,1]])*normed_Sigma[1,1]*max_scale, 'k--')
ax.plot(oval[0,:], oval[1,:], 'k:', alpha=0.3)
ax.set_aspect('equal')
ax.grid(True, which='both')
st.pyplot(fig)