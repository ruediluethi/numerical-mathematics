import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd



st.title('Zusammenhang zwischen der Ausgleichsgerade und des Korrelationskoeffizient')

st.write(r'''
  Seien $x_1, ..., x_n$ und $y_1, ..., y_n$ zwei Datenreihen.
  So berechnen sich die Koeffizienten $a, b$ der [Ausgleichsgeraden](https://de.wikipedia.org/wiki/Methode_der_kleinsten_Quadrate#Lineare_Modellfunktion) $y = a + bx$ durch die Normalengleichung: 
''')
st.latex(r'''
    g(a,b) = \left\lVert A \left(\begin{array}{c}
         a \\ b
    \end{array}\right) - y \right\rVert_2^2 \rightarrow \min
    \quad \Rightarrow \quad
    g'(a,b) = \underbrace{A^\top A\left(\begin{array}{c}
         a \\ b
    \end{array}\right) - A^\top y}_{\text{Normalengleichung}} \stackrel{!}{=} 0 \\
    \text{mit} \quad A = \left(\begin{array}{cc}
      1 & x_1 \\
      \vdots & \vdots \\
      1 & x_n \\
    \end{array}\right)
    \quad \text{und} \quad y = \left(\begin{array}{c}
      y_1 \\
      \vdots \\
      y_n \\  
    \end{array}\right)
''')
st.write(r'''Durch ausmultiplizieren von $A^\top A$ und $A^\top y$ folgt''')
st.latex(r'''
    A^\top A\left(\begin{array}{c}
         a \\ b
    \end{array}\right) - A^\top y = 0
    \quad \Leftrightarrow \quad
    \underbrace{\left(\begin{array}{cc}
      \sum_{i=1}^n 1 = n & \sum_{i=1}^n x_i \\
      \sum_{i=1}^n x_i & \sum_{i=1}^n x_i^2 \\
    \end{array}\right)}_{A^\top A} \left(\begin{array}{c}
         a \\ b
    \end{array}\right) = \underbrace{\left(\begin{array}{c}
      \sum_{i=1}^n y_i \\
      \sum_{i=1}^n x_i y_i \\
    \end{array}\right)}_{A^\top y}
''')
st.write(r'''
  Ausgeschrieben als Gleichungssystem:
''')
st.latex(r'''
  a n + b \sum_{i=1}^n x_i = \sum_{i=1}^n y_i 
  \quad\Rightarrow\quad a = \sum_{i=1}^n \frac{y_i}{n} - b \sum_{i=1}^n \frac{x_i}{n} 
  \quad \text{(1)} \\
  a \sum_{i=1}^n x_i + b \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i
  \quad\stackrel{\text{(1)}}{\Rightarrow}\quad
  \left(\sum_{i=1}^n \frac{y_i}{n} - \sum_{i=1}^n \frac{x_i}{n}\right) \sum_{i=1}^n \frac{x_i}{n} + b \sum_{i=1}^n \frac{x_i^2}{n} = \sum_{i=1}^n \frac{x_i y_i}{n} \\
  \Rightarrow\quad b \underbrace{\left( \sum_{i=1}^n \frac{x_i^2}{n} - \left( \sum_{i=1}^n \frac{x_i}{n} \right)^2 \right)}
  _{\mathbb{E(X^2)} - \mathbb{E}(X)^2 = \mathbb{E}\left( \left( X - \mathbb{E}(X) \right)^2 \right) = \text{Var}(X)}
  + \underbrace{\sum_{i=1}^n \frac{x_i}{n}}_{\mathbb{E}(X)} \underbrace{\sum_{i=1}^n \frac{y_i}{n}}_{\mathbb{E}(Y)} 
  = \underbrace{\sum_{i=1}^n \frac{x_i y_i}{n}}_{\mathbb{E}(XY)} \\
  \Leftrightarrow\quad b = \frac{\mathbb{E}(XY) - \mathbb{E}(X) \mathbb{E}(Y)}{\text{Var}(X)}
  = \frac{\mathbb{E} \left( \left(X - \mathbb{E}(X) \right) \left(Y - \mathbb{E}(Y) \right) \right)}{\text{Var}(X)}
  = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}
''')
st.write(r'''
  Werden zusätzlich die Datenreihen $x_1, ..., x_n$ und $y_1, ..., y_n$ mit deren Standardabweichung $\sigma = \sqrt{\text{Var}(X)}$ normiert,
  so ergibt sich der [Korrelationskoeffizient](https://de.wikipedia.org/wiki/Korrelationskoeffizient_nach_Bravais-Pearson#Definition) $r$:
''')
st.latex(r'''
  \frac{\text{Cov}\left(\frac{X}{\sigma_X},\frac{Y}{\sigma_Y}\right)}{\text{Var}\left(\frac{X}{\sigma_X}\right)}
  = \frac{\mathbb{E} \left( \left(\frac{X}{\sigma_X} - \mathbb{E}\left(\frac{X}{\sigma_X}\right) \right) 
  \left(\frac{Y}{\sigma_Y} - \mathbb{E}\left(\frac{Y}{\sigma_Y}\right) \right) \right)}{\frac{1}{\sigma_X^2} \text{Var}(X)} \\
  = \frac{\mathbb{E} \left( \frac{1}{\sqrt{\text{Var}(X)} \sqrt{\text{Var}(Y)}}
  \left(X - \mathbb{E}(X) \right) \left(Y - \mathbb{E}(Y) \right)\right)}
  {\frac{1}{\sqrt{\text{Var}(X)}^2 } \text{Var}(X)}
  = \frac{\mathbb{E} \left( 
  \left(X - \mathbb{E}(X) \right) \left(Y - \mathbb{E}(Y) \right)\right)}
  {\sqrt{\text{Var}(X)} \sqrt{\text{Var}(Y)}} = r
''')

df = pd.read_csv('./data/korrelation_example.csv')
# st.write(df)




x = df['impulse'].values
y = df['x1'].values

example = st.radio('Example', ['Kraftimpuls (X) VS Werkzeugverschleiß (Y)', 'Gleichverteilte Zufallszahlen'], horizontal=True)

if example == 'Gleichverteilte Zufallszahlen':
  spread = st.slider('spread', 0.0, 1.0, 0.5)
  if st.button('reset'):
    del st.session_state.x
    del st.session_state.y

  if 'x' in st.session_state:
    x	= st.session_state.x
  else:
    x = np.linspace(0, 1, 100) + np.random.rand(100)*spread
    st.session_state.x = x

  if 'y' in st.session_state:
    y	= st.session_state.y
  else:
    y = np.linspace(0, 2, 100) + np.random.rand(100)*spread
    st.session_state.y = y

if st.checkbox('Mit Standardabweichung normieren'):
  x = x / np.std(x)
  y = y / np.std(y)

if st.checkbox('X und Y vertauschen'):
  temp = x
  x = y
  y = temp

# x = (x - np.min(x))/(np.max(x) - np.min(x))
# y = (y - np.min(y))/(np.max(y) - np.min(y))
fig, ax = plt.subplots()
ax.plot(x, label='X')
ax.plot(y, label='Y')
ax.legend()
st.pyplot(fig)

A = np.ones((x.size, 2))
A[:,1] = x

coefs = np.linalg.solve(A.T @ A, A.T @ y)

df_vars = pd.DataFrame(columns=['Variable', 'Wert (Formel)', 'Wert (Python)'])
df_vars = pd.concat([df_vars, pd.DataFrame([{
    'Variable': 'Steigung der Ausgleichsgerade [b]',
    'Wert (Formel)': coefs[1],
    'Wert (Python)': '-'
}])], ignore_index=True)

fig, ax = plt.subplots()
ax.plot(x, y, '.', label='Datenpunkte')
ax.plot(x, A @ coefs, label='Ausgleichsgerade')
st.pyplot(fig)

n = x.size
cov_value = np.mean((x - np.mean(x)) * (y - np.mean(y)))
r_value = cov_value / np.sqrt(np.var(x) * np.var(y))

r = np.corrcoef(x, y)
cov = np.cov(x, y)


df_vars = pd.concat([df_vars, pd.DataFrame([{
    'Variable': 'Kovarianz [Cov(X,Y)]',
    'Wert (Formel)': cov_value,
    'Wert (Python)': cov[0,1]
}])], ignore_index=True)

df_vars = pd.concat([df_vars, pd.DataFrame([{
    'Variable': 'Korrelationskoeffizient  [r]',
    'Wert (Formel)': r_value,
    'Wert (Python)': r[0,1]
}])], ignore_index=True)

st.markdown(df_vars.to_html(escape=False, index=False), unsafe_allow_html=True)