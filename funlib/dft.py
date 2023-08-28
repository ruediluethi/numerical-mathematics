import numpy as np
import math


def discrete_fourier_transformation(f, duration):
    n = f.size

    A = np.zeros((n,n))
    B = np.zeros((n,n))

    for k in range(0,n):
        for i in range(0,n):
            A[i][k] = 2/n*math.cos(2*math.pi*k*i/n)
            B[i][k] = -2/n*math.sin(2*math.pi*k*i/n)

    f = f.reshape((n,1))
    a = A @ f # real part
    b = B @ f # imaginary part

    frequencies = np.zeros((math.floor(n/2),5))
    # for (let i = 3/*Math.floor(n*0.01)*/; i < n/2; i++){
    for i in range(0,math.floor(n/2)):
        frequency = i/duration * 1000
        norm = math.sqrt( a[i][0]*a[i][0] + b[i][0]*b[i][0] )
        frequencies[i][0] = frequency
        frequencies[i][1] = norm
        frequencies[i][2] = a[i][0]
        frequencies[i][3] = b[i][0]
        frequencies[i][4] = math.atan2(b[i][0],a[i][0])

    return frequencies

def hamming_window(f):
  n = f.size
  alpha = 25/46
  beta = 1 - alpha
  g = np.zeros((n,1))
  for i in range(0, f.size):
    g[i] = f[i] * (alpha - beta * math.cos( 2 * math.pi * i / n ))
  return g