import streamlit as st
import numpy as np
import math 
from matplotlib import pyplot as plt

from funlib.poly import mitternacht

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

grid_res = 200

fig, ax = plt.subplots()
for i in range(0,1):
    # A = np.array([1.0, -0.8, -0.8, 3.0]).reshape(2,2)

    # A = np.array([1.0, 0, 0, 1.0]).reshape(2,2)

    A = np.random.rand(2,2)-0.5
    A = A.T @ A


    # b = np.array([1, 2]).reshape(2,1)
    b = np.array([0, 0]).reshape(2,1)

    x_sol = np.linalg.solve(A, b)

    res = 1000
    phi = np.linspace(0, 2*math.pi, res)
    circle = np.zeros((2, res))
    for i in range(0, res):
        c = math.cos(phi[i])
        s = math.sin(phi[i])
        
        r = math.sqrt(2/(A[0,0]*c*c + (A[0,1]+A[1,0])*c*s + A[1,1]*s*s))

        circle[0,i] = r*c
        circle[1,i] = r*s

    grid_size = max(np.amax(circle[0,:]) - np.amin(circle[0,:]),
                    np.amax(circle[1,:]) - np.amin(circle[1,:])) * 1

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_sol[0]-grid_size, x_sol[0]+grid_size, grid_res), 
        np.linspace(x_sol[1]-grid_size, x_sol[1]+grid_size, grid_res)
    )

    field_u, field_v = np.meshgrid(np.zeros(grid_res), np.zeros(grid_res))

    fx = np.zeros((grid_res, grid_res))

    for i in range(0,grid_res):
        for j in range(0,grid_res):
            x_ij = np.array([grid_x[i,j], grid_y[i,j]]).reshape(2,1)
            grad_ij = A @ x_ij - b
            field_u[i,j] = grad_ij[0] / np.linalg.norm(grad_ij)
            field_v[i,j] = grad_ij[1] / np.linalg.norm(grad_ij)
            fx_ij = 1/2 * x_ij.T @ A @ x_ij - b.T @ x_ij
            cost_value = fx_ij -1
            # if cost_value < 0:
            #     cost_value = cost_value * 7
            fx[i,j] = cost_value

    for i in range(0,grid_res):
        for j in range(0,grid_res):
            if fx[i,j] > 0:
                fx[i,j] = math.pow(1 - fx[i,j] / np.amax(fx), 10)
            else:
                fx[i,j] = 1 + fx[i,j]

            # fx[i,j] = math.pow(1 - fx[i,j], 3)


    st.write(np.amax(fx))

    img_extent = [
        (x_sol[0]-grid_size)[0], 
        (x_sol[0]+grid_size)[0], 
        (x_sol[1]-grid_size)[0], 
        (x_sol[1]+grid_size)[0]
    ]

    ax.set_title(r'f(x)')
    ax.imshow(np.flip(fx,0), extent=img_extent, cmap='Greys')
    # ax.quiver(grid_x, grid_y, field_u, field_v, color='grey', label=r'Gradient $\nabla f(x)$')
    # ax.contour(grid_x, grid_y, f_x, 9, colors='black')
    ax.plot(x_sol[0,:], x_sol[1,:], 'kx', label=r'$A^{-1}b$')
    # ax.plot(circle[0,:], circle[1,:], 'r')
    # ax.axis('scaled')
    # ax.legend()
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)

    i = st.slider('i', 0, res, 0)

    c = math.cos(phi[i])
    s = math.sin(phi[i])
    
    a_dia = A[0,1]+A[1,0]
    r = math.sqrt(2/(A[0,0]*c*c + a_dia*c*s + A[1,1]*s*s))
    dr = (4*A[0,0]*s*c - 4*A[1,1]*s*c + 2*a_dia*(s*s - c*c)) / math.pow(s* (A[1,1]*s + a_dia*c) + A[0,0]*c*c, 2)
    dr = dr/(2*math.sqrt(r))

    c_i = np.array([r*c, r*s]).reshape(2,1)
    dc_i = np.array([c*dr - r*s, s*dr + r*c])
    dc_i_normed = dc_i / np.linalg.norm(dc_i)

    # ax.plot(c_i[0], c_i[1], 'ko')
    # ax.plot([c_i[0], c_i[0] + dc_i_normed[0]], [c_i[1], c_i[1] + dc_i_normed[1]], 'k-')

    for j in range(0,1):
        # x_0 = np.array([1.6,0.3]).reshape(2,1)
        x_0 = (np.random.rand(2,1)-0.5)*grid_size*2
        r_0 = - A @ x_0

        r_0_normed = r_0 / np.linalg.norm(r_0)
        

        lambda_min = (r_0.T @ r_0) / (r_0.T @ A @ r_0)
        
        lambda_1, lambda_2 = mitternacht(1/2 * r_0.T @ A @ r_0, 
                                        r_0.T @ A @ x_0,
                                        1/2 * x_0.T @ A @ x_0 - 1)

        if (lambda_1 == 0 or lambda_2 == 0):
            # ax.plot(x_0[0], x_0[1], 'ro')
            break

        lambda_dist = min(lambda_1, lambda_2)

        x_end = x_0 + r_0*lambda_dist

        
        
        # ax.plot(x_0[0], x_0[1], 'k.')
        # ax.plot([x_0[0], x_end[0]], [x_0[1], x_end[1]])
        ax.set_aspect(1)

        step_res = 100
        steps = np.linspace(0, lambda_min[0] * 1.5, step_res)
        F_of_lambda = np.zeros(step_res)
        for i in range(0,step_res):
            x_lambda = x_0 + steps[i] * r_0
            F_of_lambda[i] = 1/2 * x_lambda.T @ A @ x_lambda

        x_lambda_min = x_0 + lambda_min * r_0
        F_of_lambda_min = 1/2 * x_lambda_min.T @ A @ x_lambda_min

        x_lambda_1 = x_0 + lambda_1 * r_0
        F_of_lambda_1 = 1/2 * x_lambda_1.T @ A @ x_lambda_1
        x_lambda_2 = x_0 + lambda_2 * r_0
        F_of_lambda_2 = 1/2 * x_lambda_2.T @ A @ x_lambda_2



st.pyplot(fig)




# fig, ax = plt.subplots()
# ax.plot(0, 0, 'k+')

# ok = False
# while ok is False:
#     q = np.random.rand()-0.5
#     h = np.random.rand()-0.5
#     p = np.random.rand()-0.5
#     # q = 1
#     # h = 1
#     # p = 1

    
#     res = 1000
#     phi = np.linspace(0, 2*math.pi, res)
#     circle = np.zeros((2, res))
#     for i in range(0, res):
#         c = math.cos(phi[i])
#         s = math.sin(phi[i])
#         d = q*c*c + h*c*s + p*s*s
#         if d < 0:
#             ok = False
#             break
#         else:
#             ok = True

#         r = math.sqrt(1/d)

#         circle[0,i] = r*c
#         circle[1,i] = r*s

#     if ok:

#         a = 4*q*p - h*h
#         b = 4*q + 4*p
#         c = 4

#         lambda_1 = (-b + math.sqrt(b*b - 4*a*c))/(2*a)
#         lambda_2 = (-b - math.sqrt(b*b - 4*a*c))/(2*a)

#         lambda_1_ = (1 + lambda_1 * q)/(lambda_1 * h)
#         lambda_2_ = (1 + lambda_2 * q)/(lambda_2 * h)

#         st.write(lambda_1_, lambda_2_)

#         x_1 = math.sqrt(1/(q - 2*h*lambda_1_ + 4*p*lambda_1_*lambda_1_))
#         x_2 = math.sqrt(1/(q - 2*h*lambda_2_ + 4*p*lambda_2_*lambda_2_))

#         y_1 = -2*x_1*lambda_1_
#         y_2 = -2*x_2*lambda_2_

#         ax.plot(circle[0,:], circle[1,:])

#         ax.plot(x_1, y_1, '.')
#         ax.plot(-x_1, -y_1, '.')
#         ax.plot(x_2, y_2, '.')
#         ax.plot(-x_2, -y_2, '.')
# ax.set_aspect(1)
# st.pyplot(fig)


# st.write('hallo')