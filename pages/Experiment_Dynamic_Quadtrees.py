import streamlit as st
import numpy as np
import math 
from matplotlib import pyplot as plt

from funlib.poly import mitternacht
from funlib.quad_square import QuadSquare, line_intersection

print('--- rerun ---')


x_11, y_11 = 0.472, 0.4
x_12, y_12 = 0.5+0.12, 0.1
x_21, y_21 = 0.5-0.12, 0.1
x_22, y_22 = 0.5+0.12, 0.5

x, y = line_intersection(x_11, y_11, x_12, y_12, x_21, y_21, x_22, y_22)

fig, ax = plt.subplots()
ax.plot([x_11, x_12], [y_11, y_12], 'r')
ax.plot([x_21, x_22], [y_21, y_22], 'b')
ax.plot(x, y, 'gX')
st.write(x, y)
st.pyplot(fig)


# st.stop()


st.subheader('Kreis als Funktion von x')
st.write(r'''
    Ein Kreis ist im euklidischen Raum definiert als der Rand einer Menge,
    welche zum Ursprung den gleichen Abstand $r$ besitzt
''')
st.latex(r'''
    \left\Vert \left( \begin{array}{c}
        x \\
        y
    \end{array} \right) \right\Vert_2^2
    = x^2 + y^2 = r^2
    \quad \Leftrightarrow \quad y = \sqrt{r^2 - x^2}
''')
st.write(r'mit Verschiebung zu einem definierten Kreismittelpunkt $c$')
st.latex(r'''
    \left\Vert \left( \begin{array}{c}
        x - c_x \\
        y - c_y
    \end{array} \right) \right\Vert_2^2
    = (x - c_x)^2 + (y - c_y)^2 = r^2
    \quad \Leftrightarrow \quad y = \sqrt{r^2 - (x - c_x)^2} + c_y = f(x)
''')

st.write(r'Erste Ableitung')
st.latex(r'''
    f'(x) = \frac{1}{2} \cdot \frac{1}{\sqrt{r^2 - (x - c_x)^2}} \cdot -2(x - c_x) = -\frac{x - c_x}{\sqrt{r^2 - (x - c_x)^2}}     
''')
def df(x, c_x, r):
    inside_sqrt = r**2 - (x - c_x)**2
    if inside_sqrt < 0:
        return 0
    return -(x - c_x)/np.sqrt(inside_sqrt)


st.write(r'Für den Schnittpunkt mit einer Geraden $y = a + bx$ gilt')
st.latex(r'''
    \sqrt{r^2 - (x - c_x)^2} + c_y = a + bx \\
    \Leftrightarrow \quad r^2 - (x - c_x)^2 = (a + bx - c_y)^2 \\
    \Leftrightarrow \quad r^2 - x^2 + 2x c_x - c_x^2 = b^2 x^2 + 2bx(a - c_y) + (a-c_y)^2 \\
    \Leftrightarrow \quad x^2(1 + b^2) + x\left(2b\left(a - c_y\right) - 2 c_x\right) - r^2 + c_x^2 + (a-c_y)^2 = 0 
''')

st.write(r'Für den Schnittpunkt mit einer Horizontalen $h$ gilt')
st.latex(r'''
    \sqrt{r^2 - (x - c_x)^2} + c_y = h \\
    r^2 - (h - c_y)^2 = (x - c_x)^2 = x^2 - 2x c_x + c_x^2 \\
    x^2 - 2x c_x + c_x^2 - r^2 + (h - c_y)^2 = 0
''')

def plot_circle(ax, c_x, c_y, r, resolution=100, alpha=1.0):
    phi = np.linspace(0, 2*np.pi, resolution)
    ax.plot(np.cos(phi)*r + c_x, np.sin(phi)*r + c_y, 'r', alpha=alpha)

    # x = np.linspace(c_x-r, c_x+r)
    # y = np.sqrt(r**2 - (x - c_x)**2) + c_y
    # ax.plot(x, y, 'k.', alpha=0.3)

def plot_line(ax, a, b, resolution=100):
    x = np.linspace(-1, 1, resolution)
    ax.plot(x, a + b*x)


fig, ax = plt.subplots()
ax.set_aspect('equal')

# circle
c_x = 0.33
c_y = 0.46
r = 0.2
plot_circle(ax, c_x, c_y, r)

# line
a = 0.3
b = 0.8
plot_line(ax, a, b)

x_1, x_2 = mitternacht(1+b**2, 2*b*(a - c_y) - 2*c_x, -r**2 + c_x**2 + (a - c_y)**2)
ax.plot(x_1, a + b*x_1, 'gX')
ax.plot(x_2, a + b*x_2, 'gX')

# vertical line
h = 0.7
ax.plot([c_x-r, c_x+r], [h, h])
x_1, x_2 = mitternacht(1, -2*c_x, c_x**2 - r**2 + (h - c_y)**2)

ax.plot(x_1, h, 'rX')
ax.plot(x_2, h, 'rX')

dx_1 = df(x_1, c_x, r)
x = np.linspace(x_1-r/2, x_1+r/2, 100)
ax.plot(x, dx_1*(x - x_1) + h, 'r--')

st.pyplot(fig)

grid_width = 1.0
grid_height = 1.0
square_size = 0.5


# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# plot_circle(ax, c_x, c_y, r)
# for (x, y) in [(0.472, 0.4)]:#, (0.5+0.12, 0.1)]:
# # for x in np.linspace(square_size/2, grid_width-square_size/2, round(grid_width/square_size)):
# #     for y in np.linspace(square_size/2, grid_height-square_size/2, round(grid_height/square_size)):
#         s = QuadSquare(x, y, 1.0/2/2)
#         s.plot(ax, alpha=1.0)

#         st.write(x, y, s.calc_circle_intersection(c_x, c_y, r, ax=ax))


# st.pyplot(fig)


grid_size = 1.0
split_threshold = 0.02
fig, ax = plt.subplots()
ax.set_aspect('equal')
s = QuadSquare(grid_size/2, grid_size/2, grid_size)

res = 5
i = 0
for phi in np.linspace(0.0, np.pi*1/2, res):
    x = c_x + np.cos(phi)*0.5
    y = c_y + np.sin(phi)*0.5
    s.calc_circle_intersection(x, y, r, split_threshold=split_threshold)#, ax=ax)
    plot_circle(ax, x, y, r, 100, alpha=0.2)
    i = i + 1
    # if i >= 2:
    #     break

for phi in np.linspace(np.pi, np.pi*3/2, res):
    x = c_x+0.35 + np.cos(phi)*0.5
    y = c_y+0.35 + np.sin(phi)*0.5
    s.calc_circle_intersection(x, y, r, split_threshold=split_threshold)#, ax=ax)
    plot_circle(ax, x, y, r, 100, alpha=0.2)

s.plot(ax)

st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_aspect('equal')


s = QuadSquare(grid_size/2, grid_size/2, grid_size)
s.calc_circle_intersection(c_x, c_y, r, split_threshold=split_threshold)#, ax=ax)
s.plot(ax)

s = QuadSquare(grid_size/2, grid_size/2, grid_size)
s.calc_circle_intersection(c_x+0.05, c_y+0.18, r, split_threshold=split_threshold)#, ax=ax)
s.plot(ax)

s.plot(ax)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_aspect('equal')
s = QuadSquare(grid_size/2, grid_size/2, grid_size)
s.calc_circle_intersection(c_x, c_y, r, split_threshold=split_threshold)#, ax=ax)
# s.calc_circle_intersection(c_x-0.1, c_y+0.25, r, split_threshold=split_threshold, ax=ax)
s.plot(ax)

# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)

st.pyplot(fig)

fig, ax = plt.subplots()
ax.set_aspect('equal')
s = QuadSquare(grid_size/2, grid_size/2, grid_size)
s.calc_circle_intersection(c_x, c_y, r, split_threshold=split_threshold)#, ax=ax)
s.calc_circle_intersection(c_x-0.1, c_y+0.25, r, split_threshold=split_threshold)#, ax=ax)
s.plot(ax)

# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)

st.pyplot(fig)
