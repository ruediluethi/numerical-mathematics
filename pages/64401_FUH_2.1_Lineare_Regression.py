from funlib.pca import PCA
import streamlit as st
import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from datafun.lego import get_X_color_for_set_names, load_lego_data, select_examples

st.title('Lineare Regression')

st.page_link('pages/data_lego.py', label='Datenquelle: Lego Database')

D = load_lego_data()
df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D

set_names_A, set_names_B, min_parts = select_examples(D)

X_A, X_list_A = get_X_color_for_set_names(D, set_names_A, min_parts=min_parts)
X_B, X_list_B = get_X_color_for_set_names(D, set_names_B, min_parts=min_parts)


plot_colors_A = ['red', 'purple', 'orange']
plot_colors_B = ['blue', 'green', 'cyan']


X = np.vstack((X_A, X_B))
A = PCA(X, d=3)
A_A = np.zeros((X_A.shape[0], A.shape[1]))
A_B = np.zeros((X_B.shape[0], A.shape[1]))

fig, ax = plt.subplots()
k = 0
for i, X_ in enumerate(X_list_A):
    k_next = k + X_.shape[0]
    ax.plot(A[k:k_next,0], A[k:k_next,1], 'o', alpha=0.5, label=set_names_A[i], color=plot_colors_A[i%len(plot_colors_A)])
    A_A[k:k_next,:] = A[k:k_next,:]
    k = k_next
k_B = 0
for i, X_ in enumerate(X_list_B):
    k_next = k + X_.shape[0]
    k_B_next = k_B + X_.shape[0]
    ax.plot(A[k:k_next,0], A[k:k_next,1], 'o', alpha=0.5, label=set_names_B[i], color=plot_colors_B[i%len(plot_colors_B)])
    A_B[k_B:k_B_next,:] = A[k:k_next,:]
    k = k_next
    k_B = k_B_next

ax.set_xlabel('1. Hauptachse')
ax.set_ylabel('2. Hauptachse')

# ax.plot(A_A[:,0], A_A[:,1], '.', label='Group A', color='red')
# ax.plot(A_B[:,0], A_B[:,1], '.', label='Group B', color='blue')

st.write('''
    Für jedes Legeo-Set wird die Anzahl der Steine pro Farbe gezählt und 
    durch die gesamt Anzahl der Steine eines Sets geteilt.
    So wird jedes Set durch einen Zeilenvektor $x_i$ repräsentiert,
    welcher als Merkmale die jeweilige prozentualen Anteile der Steinfarbe enthält.
''')

ax.legend()
st.pyplot(fig)
st.caption('''
    Die Lego-Sets aus den gewählten Themen werden mittels PCA auf die zwei (respektive drei) Hauptachsen projiziert
    und dem Thema entsprechend eingefärbt.
''')

st.write(r'''
    Die Matrix $X \in \mathbb{R}^{n \times D}$ enthält die Datenpunkte von $n$ Objekten (Zeilenvektor $x_i$) 
    mit $D$ Merkmalen/Dimensionen (Spalten - in diesem Falle die Hauptachsen).
''')

st.latex(r'''
    X = \left(\begin{array}{cccc}
        x_{11} & x_{12} & \cdots & x_{1D} \\
        \vdots & & \ddots & \vdots \\
        x_{n1} & x_{12} & \cdots & x_{nD} \\
    \end{array}\right)
''')

st.write(r'''
    Nun soll ein linearer Klassifikator $f$ mit den Gewichten $w \in \mathbb{R}^{D+1}$ bestimmt werden, 
    welcher die Daten der Matrix $X$ möglichst gut auf einen Zielvektor $y \in \mathbb{R}^n$ approximiert. 
    Damit der Klassifikator nicht nur skaliert, sondern auch eine Translation berücksichtigt, wird die Datenmatrix $X$ um eine Spalte erweitert
    und die Approximation kann als Matrixmultiplikation beschrieben werden:
''')

st.latex(r'''
    \left(\begin{array}{cccc}
        1 & x_{11} & x_{12} & \cdots & x_{1D} \\
        1 & \vdots & & \ddots & \vdots \\
        1 & x_{n1} & x_{12} & \cdots & x_{nD} \\
    \end{array}\right)
    \left(\begin{array}{c}
        w_0 \\
        \vdots \\
        w_D
    \end{array}\right) = \tilde{X}w
''')

st.write(r'''
    Nun sollen die Gewichte $w$ so gewählt werden, dass folgende Fehlerfunktion $J(w)$ minimal wird.      
''')

st.latex(r'''
    \left( y - \tilde{X}w \right)^\intercal  \left( y - \tilde{X}w \right)
    = \sum_{i=1}^{n} \left( y_i - \tilde{x}_i w \right)^2
    = J(w)
''')

st.write(r'''
    Die Fehlerfunktion $J(w)$ ist dann Minimal, wenn ihre Ableitung nach den Gewichten $w$ Null ist.
''')
st.latex(r'''
    \frac{\partial J(w)}{\partial w}
    = \frac{\partial}{\partial w} ( y^\intercal y \overbrace{- y^\intercal \tilde{X} w - w^\intercal \tilde{X}^\intercal y}^{
        \textrm{da } y^\intercal \tilde{X} w, w^\intercal \tilde{X}^\intercal y \in \mathbb{R} \quad \Rightarrow \quad 2w^\intercal \tilde{X}^\intercal y
    } + w^\intercal \tilde{X}^\intercal \tilde{X} w ) \\
    = 2 \tilde{X}^\intercal \tilde{X} w - 2 \tilde{X}^\intercal y
    \stackrel{!}{=} 0 \\
    \Leftrightarrow\quad w = \left( X^\intercal X \right)^{-1} X^\intercal y \\
    \Leftrightarrow\quad \underbrace{X^\intercal X}_{=A}  w = \underbrace{X^\intercal y}_{b}
''')

st.write(r'''
    Die Gewichte $w$ können nun durch das Lösen des Gleichungssystem $Aw = b$ bestimmt werden
''')

st.write(r'''
    Um nun einen neuen Datenpunkt $x_{\textrm{neu}}$ zu klassifizieren, muss dieser bloß mit den Gewichten $w$ multipliziert werden.
''')
st.latex(r'''
    f(x_{\textrm{neu}}) = \left( 1, x_{\textrm{neu},1}, ..., x_{\textrm{neu},D} \right) w = \hat{y}
''') 

st.write(r'''
    Im 2D Fall entspricht die Entscheidunggrenze der linearen Klassifikation einer Geradengleichung 
    und der Wert g der rechten Seite ist gerade das Entscheidungskriterium. 
''')
st.latex(r'''
    x^\top w = w_0 + w_1 x_1 + w_2 x_2 = g
''')


def classify(X_a, X_b):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 3))
    X[0:X_a.shape[0],1:3] = X_a
    X[X_a.shape[0]:,1:3] = X_b

    y = np.ones(n)
    y[X_a.shape[0]:] = -1

    # pseudo_inv = linalg.inv(X.T @ X) @ X.T
    # w = pseudo_inv @ y
    # st.write(w)

    w = linalg.solve(X.T @ X, X.T @ y)
    # st.write(w)

    fig, ax = plt.subplots()
    ax.plot(X[:,1][y==1], X[:,2][y==1], 'wo', markeredgecolor='k', label='Group A')
    ax.plot(X[:,1][y==-1], X[:,2][y==-1], 'k.', label='Group B')

    x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)
    border = (-w[0]-w[1]*x) / w[2]
    border_below = np.argwhere((X[:,2].min() < border) & (border < X[:,2].max()))
    ax.plot(x[border_below], border[border_below], 'k--', label='decision boundary')
    
    X_x_range = np.amax(X[:,1]) - np.amin(X[:,1])
    ax.set_xlim([np.amin(X[:,1]) - X_x_range*0.05, np.amax(X[:,1]) + X_x_range*0.05])
    X_y_range = np.amax(X[:,2]) - np.amin(X[:,2])
    ax.set_ylim([np.amin(X[:,2]) - X_y_range*0.05, np.amax(X[:,2]) + X_y_range*0.05])

    ax.legend()
    st.pyplot(fig)

classify(A_A[:,0:2], A_B[:,0:2])


st.write(r'''
    Die Trennlinie kann auch ein Polynom zweiten Grades sein.     
''')
st.latex(r'''
    X = \left(\begin{array}{ccccccc}
        x_{11} & x_{11}^2 & x_{11} & x_{11}^2 & \cdots & x_{1D} & x_{1D}^2 \\
        \vdots & & \ddots & \vdots \\
        x_{n1} & x_{n1}^2 & x_{12} & x_{12}^2 & \cdots & x_{nD} & x_{nD}^2 \\
    \end{array}\right)
''')


def classify_poly(X_a, X_b):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 3))
    X[0:X_a.shape[0],1:3] = X_a
    X[X_a.shape[0]:,1:3] = X_b

    X2 = np.ones((n, 5))
    X2[:,1] = X[:,1]
    X2[:,2] = X[:,1]**2
    X2[:,3] = X[:,2]
    X2[:,4] = X[:,2]**2

    # print(X2)

    y = np.ones(n)
    y[X_a.shape[0]:] = -1

    pseudo_inv = linalg.inv(X2.T @ X2) @ X2.T
    w = pseudo_inv @ y

    fig, ax = plt.subplots()
    ax.plot(X[:,1][y==1], X[:,2][y==1], 'wo', markeredgecolor='k', label='Group A')
    ax.plot(X[:,1][y==-1], X[:,2][y==-1], 'k.', label='Group B')

    x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)

    c = w[0] + w[1]*x + w[2]*x**2
    b = w[3]
    a = w[4]

    ax.plot(x, (-b + np.sqrt(b**2 - 4*a*c)) /(2*a), 'k--', label='decision boundary')
    ax.plot(x, (-b - np.sqrt(b**2 - 4*a*c)) /(2*a), 'k--')

    X_x_range = np.amax(X2[:,1]) - np.amin(X2[:,1])
    ax.set_xlim([np.amin(X2[:,1]) - X_x_range*0.05, np.amax(X2[:,1]) + X_x_range*0.05])
    X_y_range = np.amax(X2[:,3]) - np.amin(X2[:,3])
    ax.set_ylim([np.amin(X2[:,3]) - X_y_range*0.05, np.amax(X2[:,3]) + X_y_range*0.05])
    ax.legend()
    st.pyplot(fig)

classify_poly(A_A[:,0:2], A_B[:,0:2])



st.write(r'''
    Im 3D Fall entspricht die Entscheidunggrenze der linearen Klassifikation einer Ebene:
''')

def classify_3d(X_a, X_b):
    n = X_a.shape[0] + X_b.shape[0]
    X = np.ones((n, 4))
    X[:X_a.shape[0],1:4] = X_a
    X[X_a.shape[0]:,1:4] = X_b

    y = np.ones(n)
    y[X_a.shape[0]:] = -1


    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:X_a.shape[0],1], y=X[:X_a.shape[0],2], z=X[:X_a.shape[0],3],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Group A'
    ))
    fig.add_trace(go.Scatter3d(
        x=X[X_a.shape[0]:,1], y=X[X_a.shape[0]:,2], z=X[X_a.shape[0]:,3],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Group B'
    ))

    pseudo_inv = linalg.inv(X.T @ X) @ X.T
    w = pseudo_inv @ y
    # st.write(w)

    m = 10
    x = np.linspace(np.amin(X[:,1]), np.amax(X[:,1]), m)
    y = np.linspace(np.amin(X[:,2]), np.amax(X[:,2]), m)
    z = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            z[j,i] = (-w[0] - w[1]*x[i] - w[2]*y[j]) / w[3]
            if z[j,i] < np.amin(X[:,3]) or z[j,i] > np.amax(X[:,3]):
                z[j,i] = np.nan

    fig.add_trace(go.Surface(
        x=x, y=y, z=z
    ))
    st.plotly_chart(fig)


classify_3d(A_A[:,[0,2,1]], A_B[:,[0,2,1]])

