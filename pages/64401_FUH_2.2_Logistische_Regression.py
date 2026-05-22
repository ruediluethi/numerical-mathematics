from funlib.pca import PCA
import streamlit as st
import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from datafun.lego import get_X_color_for_set_names, load_lego_data, select_examples

from sklearn.linear_model import LogisticRegression

st.title('Logistische Regression')

st.page_link('pages/data_lego.py', label='Datenquelle: Lego Database')


D = load_lego_data()
df_themes, df_sets, df_inventories, df_inv_parts, df_color_list, df_parts, df_part_types, themes_root, color_list, part_types_ids, part_types_names = D

set_names_A, set_names_B, min_parts = select_examples(D)

X_A, X_list_A = get_X_color_for_set_names(D, set_names_A, min_parts=min_parts)
X_B, X_list_B = get_X_color_for_set_names(D, set_names_B, min_parts=min_parts)


st.write('''
    Wir haben $n$ Datenpunkte $x_i$ in einem $d$-dimensionalen Raum.
    Dann ist die Sigmoidfunktion mit $w \in \mathbb{R}^{d+1}$ definiert durch:
''')
st.latex(r'''
    h_w(x) = \frac{1}{1 + e^{-(w_0 + w_1 x_1 + \dots + w_d x_d)}} = \hat{y}
''')


st.write('''
    Über Likelihood wird die Kostenfunktion $L$ herrgeleitet (?)
''')
st.latex(r'''
    L_{logit} = -\sum_{i=1}^{m} y_i \log \hat{y} + (1-y_i) \log(1-\hat{y})
''')
st.write('''
    Lösen des Optimierungsproblems durch Gradientenabstieg.
''')

plot_colors_A = ['red', 'purple', 'orange']
plot_colors_B = ['blue', 'green', 'cyan']

n_a = X_A.shape[0]
n_b = X_B.shape[0]
n = n_a + n_b
X = np.vstack((X_A, X_B))
d = st.slider('Anzahl der Hauptkomponenten', min_value=1, max_value=min(X.shape[1], 10), value=2, step=1)
A = PCA(X, d=d)

y = np.ones(n)
y[n_a:] = 0 

fig, ax = plt.subplots()
ax.plot(A[:n_a,0], y[:n_a], 'wo', markeredgecolor='k', label='Group A')
ax.plot(A[n_a:,0], y[n_a:], 'k.', label='Group B')


clf = LogisticRegression(max_iter=1000, C=1e3).fit(A, y.flatten())

# st.write(clf.coef_, clf.intercept_)

w = clf.coef_.flatten()
b = clf.intercept_[0]

# value_range = A[:,0].max() - A[:,0].min()

def sigmoid_1D(x):
    return 1 / (1 + np.exp(-(b + w[0] * x)))
def sigmoid(X):
    return 1 / (1 + np.exp(-(b + X @ w)))

if d == 1:
    x = np.linspace(A[:,0].min(), A[:,0].max(), 100)
    ax.plot(x, sigmoid_1D(x), 'k-', label=r'Sigmoidfunktion $h_w(x)$')


y_hat = sigmoid(A)

# positive = 

# ax.plot([x[0], x[-1]], [0.5, 0.5], 'k--', label='Entscheidungsgrenze')

true_positive = A[:n_a,:][y_hat[:n_a] >= 0.5]
false_negative = A[:n_a,:][y_hat[:n_a] < 0.5]
true_negative = A[n_a:,:][y_hat[n_a:] < 0.5]
false_positive = A[n_a:,:][y_hat[n_a:] >= 0.5]


if true_positive.size > 0:
    ax.plot(true_positive[:,0], sigmoid(true_positive), 'wo', markeredgecolor='g', label='True Positive')
if false_negative.size > 0:
    ax.plot(false_negative[:,0], sigmoid(false_negative), 'wo', markeredgecolor='r', label='False Negative')
if true_negative.size > 0:
    ax.plot(true_negative[:,0], sigmoid(true_negative), 'g.', label='True Negative')
if false_positive.size > 0:
    ax.plot(false_positive[:,0], sigmoid(false_positive), 'r.', label='False Positive')


for i in range(n):
    if i < n_a:
        if y_hat[i] >= 0.5:
            ax.plot([A[i,0], A[i,0]], [y[i], y_hat[i]], 'g:')
        else:
            ax.plot([A[i,0], A[i,0]], [y[i], y_hat[i]], 'r:')
    else:
        if y_hat[i] < 0.5:
            ax.plot([A[i,0], A[i,0]], [y[i], y_hat[i]], 'g:')
        else:
            ax.plot([A[i,0], A[i,0]], [y[i], y_hat[i]], 'r:')

ax.legend()
ax.set_xlabel('1. Hauptachse')
ax.set_ylabel('Klasse')
st.pyplot(fig)

confusion_matrix = np.array([[len(true_positive) / n_a,  len(false_positive) / n_b], 
                             [len(false_negative) / n_a, len(true_negative) / n_b]])
confusion_labels = np.array([['True Positive', 'False Positive'], 
                             ['False Negative', 'True Negative']])


fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix)
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_xticklabels([r'$y=1$', r'$y=0$'])
ax.set_yticklabels([r'$\hat{y} > 0.5$', r'$\hat{y} < 0.5$'])
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{confusion_labels[i, j]}\n{confusion_matrix[i, j]*100:.2f} %', ha='center', va='center', color='black')

ax.set_title('Confusion Matrix')
st.pyplot(fig)