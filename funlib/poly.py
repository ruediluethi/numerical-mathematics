import numpy as np
import math

def polynomial_interpolation(points):
  n = points.shape[0]

  A = np.zeros((n,n))
  b = np.zeros((n,1))
  for i in range(0,n):
    t = points[i][0]
    for j in range(0, n):
      A[i][j] = math.pow(t, j)
    
    b[i] = points[i][1]

  return np.linalg.solve(A, b)

def calc_polyline(coefs, a, b, res):
  polyline = np.zeros((res,2))
  dim = coefs.size

  for i in range(0,res):
    t = a + i/(res-1) * (b - a)
    p = 0
    for k in range(0,dim):
      p = p + coefs[k][0]*math.pow(t,k)

    polyline[i][0] = t
    polyline[i][1] = p

  return polyline

def mitternacht(a, b, c):
  inside_sqrt = b*b - 4*a*c

  if inside_sqrt <= 0:
    return 0, 0

  x_1 = (-b + math.sqrt(inside_sqrt))/(2*a)
  x_2 = (-b - math.sqrt(inside_sqrt))/(2*a)

  return x_1, x_2