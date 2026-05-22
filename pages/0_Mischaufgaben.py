import plotly.graph_objects as go

import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from io import BytesIO

# st.title('Mischaufgaben')

st.subheader('Aufgabe 40')

st.write(r'''
    Werden <span style="background-color: #ffa67a;">zwei Liter einer Sorte Spiritus</span> mit 
    <span style="background-color: #ffbfcb;">16 l einer anderen Sorte</span> gemischt
    und zu der Mischung noch <span style="background-color: #fbe883;">7.4 l Wasser</span> hinzugefügt, 
    so erhält man einen <span style="background-color: #24cccb;">50%igen Spiritus</span>.
''', unsafe_allow_html=True)


st.write('''
    <div style="font-family: KaTeX_Main, 'Times New Roman', serif; font-size: 19.36px; text-align: center;">
        <span style="background-color: #ffa67a;">2x</span> + 
        <span style="background-color: #ffbfcb;">16y</span> + 
        <span style="background-color: #fbe883;">7.4 ⋅ 0</span> = 
        <span style="background-color: #24cccb;">0.5</span>
        (
            <span style="background-color: #ffa67a;">2</span> + 
            <span style="background-color: #ffbfcb;">16</span> + 
            <span style="background-color: #fbe883;">7.4</span>
        )
        = 12.7
    </div>
''', unsafe_allow_html=True)
st.latex(r'''
    \Rightarrow \quad
    f_1(x) = \frac{1}{16} \left(12.7 - 2x\right) = y_1
''')

st.write(r'''
    Man erhält denselben <span style="background-color: #24cccb;">50%igen Spiritus</span> auch 
    durch das Mischen von <span style="background-color: #ffa67a;">6 l der ersten Sorte</span>
    mit <span style="background-color: #ffbfcb;">10 l der zweiten Sorte</span> und 
    durch Zugabe von <span style="background-color: #fbe883;">4.6 l Wasser</span>.
''', unsafe_allow_html=True)

st.write('''
    <div style="font-family: KaTeX_Main, 'Times New Roman', serif; font-size: 19.36px; text-align: center;">
        <span style="background-color: #ffa67a;">6x</span> + 
        <span style="background-color: #ffbfcb;">10y</span> + 
        <span style="background-color: #fbe883;">4.6 ⋅ 0</span> = 
        <span style="background-color: #24cccb;">0.5</span>
        (
            <span style="background-color: #ffa67a;">6</span> + 
            <span style="background-color: #ffbfcb;">10</span> + 
            <span style="background-color: #fbe883;">4.6</span>
        )
        = 10.3
    </div>  
''', unsafe_allow_html=True)
st.latex(r'''
    \Rightarrow \quad
    f_2(x) = \frac{1}{10} \left(10.3 - 6x\right) = y_2
''')

st.write(r'''
    Wie viel Prozent Spiritus enthalten die zum Mischen verwendeten Sorten?
''', unsafe_allow_html=True)

st.write('''
   <span style="background-color: #ffa67a; font-family: KaTeX_Main, 'Times New Roman', serif; font-size: 19.36px;">x</span> = Anteil Spiritus in % der ersten Sorte\\
   <span style="background-color: #ffbfcb; font-family: KaTeX_Main, 'Times New Roman', serif; font-size: 19.36px;">y</span> = Anteil Spiritus in % der zweiten Sorte      
''', unsafe_allow_html=True)

x = np.linspace(0, 1, 100)
y1 = (12.7 - 2*x)/16
y2 = (10.3 - 6*x)/10

fig, ax = plt.subplots()
ax.plot(x, y1, label=r'$f_1(x)$')
ax.plot(x, y2, label=r'$f_2(x)$')
ax.legend()
ax.set_xlabel(r'$x$ = Anteil Spiritus in % der ersten Sorte')
ax.set_ylabel(r'$y$ = Anteil Spiritus in % der zweiten Sorte')
st.pyplot(fig)

st.write(r'''
    Die Lösung liegt im Schnittpunkt der beiden Geraden, also dort, wo $f_1(x) = y_1 = y_2 = f_2(x)$ gilt:
''')
st.latex(r'''
    \Rightarrow \quad y_1 = y_2 \quad \Leftrightarrow \quad
    \frac{1}{16} \left(12.7 - 2x\right) = \frac{1}{10} \left(10.3 - 6x\right) \\
    \Leftrightarrow \quad 10 \left(12.7 - 2x\right) = 16 \left(10.3 - 6x\right) \\
    \Leftrightarrow \quad 127 - 20x = 164.8 - 96x \\
    \Leftrightarrow \quad 76x = 37.8 \\
    \Leftrightarrow \quad x = \frac{37.8}{76} \approx 0.4974
''')


st.subheader('Aufgabe 41')


st.write(r'''
    Ein <span style="background-color: #24cccb;">56%iger Spiritus</span> wird
    mit einer <span style="background-color: #ffbfcb;">zweiten </span><span style="background-color: #ffa67a;">Sorte</span> so vermengt, 
    dass eine Mischung von <span style="background-color: #fbe883;">102l Spiritus von 43%</span> entsteht.
''', unsafe_allow_html=True)

st.write('''
    <div style="font-family: KaTeX_Main, 'Times New Roman', serif; font-size: 19.36px; text-align: center; margin-bottom: 1rem;">
        <span style="background-color: #24cccb;">x</span> + 
        <span style="background-color: #ffa67a;">y</span> = 
        <span style="background-color: #fbe883;">102</span>
    </div>  
''', unsafe_allow_html=True)

st.write('''
    <div style="font-family: KaTeX_Main, 'Times New Roman', serif; font-size: 19.36px; text-align: center;">
        <span style="background-color: #24cccb;">0.56x</span> + 
        <span style="background-color: #ffbfcb;">z</span><span style="background-color: #ffa67a;">y</span> = 
        <span style="background-color: #fbe883;">102 ⋅ 0.43</span>
         = 43.86
    </div>  
''', unsafe_allow_html=True)
st.latex(r'''
    \Rightarrow \quad
    f_1(x, y) = \frac{1}{y} \left(43.86 - 0.56x\right) = z_1
''')




st.write(r'''
    Würden vom <span style="background-color: #24cccb;">56%igen Spiritus 9l weniger</span> und 
    von der <span style="background-color: #ffbfcb;">zweiten Sorte</span><span style="background-color: #ffa67a;"> 8l weniger</span> gewählt,
    so würde eine Mischung von <span style="background-color: #fbe883;">42%igem Spiritus</span> entstehen.
''', unsafe_allow_html=True)

st.write('''
    <div style="font-family: KaTeX_Main, 'Times New Roman', serif; font-size: 19.36px; text-align: center;">
        <span style="background-color: #24cccb;">0.56(x-9)</span> + 
        <span style="background-color: #ffbfcb;">z</span><span style="background-color: #ffa67a;">(y-8)</span> = 
        <span style="background-color: #fbe883;">(102-9-8) ⋅ 0.42</span>
        = 35.7
    </div>  
''', unsafe_allow_html=True)
st.latex(r'''
    \Rightarrow \quad
    f_2(x,y) = \frac{1}{y-8} \left(35.7 - 0.56\left(x-9\right) \right)
    = \frac{1}{y-8} \left(40.74 - 0.56x \right) = z_2
''')

st.write(r'''
    Wie viel Prozent hat der <span style="background-color: #ffbfcb;">Spiritus der zweiten Sorte $z$</span> und
    wie viele <span style="background-color: #ffa67a;">Liter $y$</span> davon werden für die erste Mischung verwendet?
''', unsafe_allow_html=True)




res = 100
V_min = 20
V_max = 100
x, y = np.meshgrid(np.linspace(V_min, V_max, res), np.linspace(V_min, V_max, res))

z_1 = 1/y*(102*0.43 - 0.56*x)
z_2 = 1/(y-8)*(35.7 - 0.56*(x-9))

x_cross = np.linspace(10, 70, res)
y_cross = 112.462 - 1.436*x_cross
z_cross = 1/y_cross*(102*0.43 - 0.56*x_cross)


col1, col2 = st.columns([1, 1])

fig = go.Figure(data=[go.Surface(x=x, y=y, z=z_1, opacity=1.0, colorscale='Viridis', showscale=False)])
fig.add_trace(go.Surface(x=x, y=y, z=z_2, opacity=1.0, showscale=False))

fig.add_trace(go.Scatter3d(x=x_cross, y=y_cross, z=z_cross, mode='lines', line=dict(color='red', width=6)))


# fig.update_layout(
#     scene=dict(
#         xaxis_title='x',
#         yaxis_title='y',
#         zaxis_title='z',
#     ),
#     title='Lösungsebene der Gleichung',
#     autosize=True,
#     margin=dict(l=0, r=0, b=0, t=40)
# )
col1.plotly_chart(fig)



x_cross = np.linspace(0, 70, res)
y_cross = 112.462 - 1.436*x_cross
y_3 = 102 - x_cross

fig, ax = plt.subplots(figsize=(3,4))
ax.plot(x_cross, y_cross, label=r'$y = 112.462 - 1.436x$')
ax.plot(x_cross, y_3, label=r'$x + y = 102$')
ax.legend()
ax.set_xlabel(r'$x$ = Liter der ersten Sorte')
ax.set_ylabel(r'$y$ = Liter der zweiten Sorte')
col2.pyplot(fig)

st.write(r"Die Lösung muss irgendwo auf der Schnittgerade $f_c(x)$ der beiden Flächen $f_1$ und $f_2$ liegen:")
st.latex(r'''
    f_1(x,y) = z_1 = z_2 = f_2(x,y) \\ 
    \quad \Leftrightarrow \quad
    \frac{1}{y} \left(43.86 - 0.56x\right) = \frac{1}{y-8} \left(40.74 - 0.56x \right) \\
    \Leftrightarrow \quad
    (y-8)\left(43.86 - 0.56x\right) = y\left(40.74 - 0.56x \right) \\
    \Leftrightarrow \quad
    43.86y - 0.56xy - 350.88 + 4.48x = 40.74y - 0.56xy \\
    \Leftrightarrow \quad
    3.12y = 350.88 - 4.48x \\
    \Leftrightarrow \quad
    y = 112.462 - 1.436x = f_c(x)
''')

st.write(r'Alle Werte für $x, y$ auf der Gerade $y = f_c(x)$ erfüllen das richtige Mischverhältnis.')

st.write(r'Die zusätzliche richtige Menge ist durch folgende Bedingung definiert:')

st.latex(r'''
    x + y = 102
    \quad \Leftrightarrow \quad
    y = 102 - x \stackrel{!}{=} 112.462 - 1.436x = f_c(x)\\
    \Leftrightarrow \quad
    1.436x - x = 0.436x = 112.462 - 102 = 10.462 \\
    \Leftrightarrow \quad x = \frac{1}{0.436} 10.462 \approx 23.995
''')