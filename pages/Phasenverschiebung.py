import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt



st.title('Phasenverschiebung')


@st.experimental_memo
def get_data():
    raw_file = os.path.join('data', 'measurement_20230523T115545335_2', 'measurement_20230523T115545335_2_0xA440.csv')  # noqa: E501

    df = pd.read_csv(raw_file, skiprows=[1])
    df.describe()

    t = np.array([])
    for index, row in df.iterrows():
        t = np.append(t, pd.Timestamp(row['time']).timestamp() * 1000)

    # fig, ax = plt.subplots()
    # ax.plot(np.diff(t))
    # st.pyplot(fig)

    bmX = df['Bm X'].to_numpy()
    bmY = df['Bm Y'].to_numpy()

    return t, bmX, bmY

t, bmX, bmY = get_data()

n_part = st.slider('n_part', 1, 3000, 1000, 1)
start_i = st.slider('start_i', 0, bmX.size, 40000, 1)
end_i = start_i + n_part

@st.experimental_memo
def get_part(start_i, end_i, t, bmX, bmY):
    fig, ax = plt.subplots()
    ax.plot(bmX)
    ax.plot(bmY)
    ax.plot([start_i, start_i], [-5, 5])
    st.pyplot(fig)

    deltaT = 1000/2500

    t_part = t[start_i:end_i]
    bmX_part = bmX[start_i:end_i]
    bmY_part = bmY[start_i:end_i]

    t_filled = np.array([])
    bmX_filled = np.array([])
    bmY_filled = np.array([])
    for i in range(1,t_part.size):
        t_i = t_part[i]
        bmX_i = bmX_part[i]
        bmY_i = bmY_part[i]
        t_prev = t_part[i-1]
        bmX_prev = bmX_part[i-1]
        bmY_prev = bmY_part[i-1]

        t_fill = t_prev + deltaT
        while t_fill < t_i - deltaT:
            t_filled = np.append(t_filled, t_fill)
            bmX_filled = np.append(bmX_filled, bmX_prev)
            bmY_filled  = np.append(bmY_filled, bmY_prev)
            t_fill = t_fill + deltaT

        t_filled = np.append(t_filled, t_i)
        bmX_filled = np.append(bmX_filled, bmX_i)
        bmY_filled = np.append(bmY_filled, bmY_i)



    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t_part, bmX_part)
    ax1.plot(t_part, bmY_part)
    ax2.plot(bmX_part, bmY_part, '.')
    ax2.axis('equal')
    st.pyplot(fig)

    
    t_part = t_filled
    bmX_part = bmX_filled
    bmY_part = bmY_filled

    fig, ax = plt.subplots()
    ax.plot(np.diff(t_part))
    st.pyplot(fig)

    return t_part, bmX_part, bmY_part


t_part, bmX_part, bmY_part = get_part(start_i, end_i, t, bmX, bmY)
n_part = t_part.size

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

    frequencies = np.zeros((math.floor(n/2),4))
    # for (let i = 3/*Math.floor(n*0.01)*/; i < n/2; i++){
    for i in range(0,math.floor(n/2)):
        frequency = i/duration * 1000
        norm = math.sqrt( a[i][0]*a[i][0] + b[i][0]*b[i][0] )
        frequencies[i][0] = frequency
        frequencies[i][1] = norm
        frequencies[i][2] = a[i][0]
        frequencies[i][3] = b[i][0]

    return frequencies

def hamming_window(f):
  n = f.size
  alpha = 25/46
  beta = 1 - alpha
  g = np.zeros((n,1))
  for i in range(0, f.size):
    g[i] = f[i] * (alpha - beta * math.cos( 2 * math.pi * i / n ))
  return g

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

@st.experimental_memo
def calc_rotation_freq(f, duration):

  fig, ax = plt.subplots()
  ax.plot(hamming_window(f))
  st.pyplot(fig)

  frequencies = discrete_fourier_transformation(hamming_window(f), duration)

  # freq = np.fft.fft(bmX_part)
  # freq_normed = np.zeros((math.floor(freq.size/2), 4))
  # for i in range(0,math.floor(freq.size/2)):
  #     frequency = i/duration * 1000
  #     norm = math.sqrt( freq.real[i]*freq.real[i] + freq.imag[i]*freq.imag[i] )
  #     freq_normed[i][0] = frequency
  #     freq_normed[i][1] = norm
  #     freq_normed[i][2] = freq.real[i]
  #     freq_normed[i][3] = freq.imag[i]

  # fig, ax = plt.subplots()
  # ax.plot(freq_normed[:,0], freq_normed[:,1])
  # ax.plot(freq_normed[:,0], freq_normed[:,2])
  # ax.plot(freq_normed[:,0], freq_normed[:,3])
  # st.pyplot(fig)

  max_i = np.argmax(frequencies[:,1])
  # max_freq = frequencies[max_i][0]

  peak_coefs = polynomial_interpolation(frequencies[max_i-1:max_i+2,[0,1]])
  peak_polyline = calc_polyline(peak_coefs, frequencies[max_i-1,0], frequencies[max_i+1,0], 100)

  real_coefs = polynomial_interpolation(frequencies[max_i-1:max_i+2,[0,2]])
#   real_polyline = calc_polyline(real_coefs, frequencies[max_i-1,0], frequencies[max_i+1,0], 100)

  imag_coefs = polynomial_interpolation(frequencies[max_i-1:max_i+2,[0,3]])
#   imag_polyline = calc_polyline(imag_coefs, frequencies[max_i-1,0], frequencies[max_i+1,0], 100)

  
  display_range = 5

  if peak_coefs.size == 0:
    return np.array([])

  # set first derivative to zero and reolve to x 
  # p(x) = a_0 + x*a_1 + x^2*a_2
  # p'(x) = a_1 + 2*x*a_2 = 0  =>  x = - a_1 / (2*a_2)
  peak_freq = - peak_coefs[1] / (2*peak_coefs[2])
  peak_value = peak_coefs[0] + peak_coefs[1]*peak_freq + peak_coefs[2]*peak_freq*peak_freq

  real_value = real_coefs[0] + real_coefs[1]*peak_freq + real_coefs[2]*peak_freq*peak_freq
  imag_value = imag_coefs[0] + imag_coefs[1]*peak_freq + imag_coefs[2]*peak_freq*peak_freq

  
  fig, ax = plt.subplots()
  ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
          frequencies[max_i-display_range:max_i+display_range,1], label=r'Norm')
  ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
          frequencies[max_i-display_range:max_i+display_range,2], label=r'Real')
  ax.plot(frequencies[max_i-display_range:max_i+display_range,0], 
          frequencies[max_i-display_range:max_i+display_range,3], label=r'Imag')
  ax.plot(peak_polyline[:,0], peak_polyline[:,1])
  ax.plot(peak_freq, peak_value, '.')
#   ax.plot(real_polyline[:,0], real_polyline[:,1])
  ax.plot(peak_freq, real_value, '.')
#   ax.plot(imag_polyline[:,0], imag_polyline[:,1])
  ax.plot(peak_freq, imag_value, '.')
  ax.legend()
  st.pyplot(fig)

  return np.array([peak_freq, peak_value, real_value, imag_value])

duration = t[end_i] - t[start_i]

peakX_freq = calc_rotation_freq(bmX_part, duration)
peakY_freq = calc_rotation_freq(bmY_part, duration)

peak_freq = 1
if peakX_freq.size > 0 & peakX_freq.size > 0:
    peak_freq = (peakX_freq[0] + peakY_freq[0]) / 2

elif peakX_freq.size > 0:
    peak_freq = peakX_freq[0]

elif peakY_freq.size > 0:
    peak_freq = peakY_freq[0]

delta_t = 1000 / peak_freq

t_cutted = np.zeros((n_part,1))
for i in range(0,n_part):
  t_cutted[i] = (t_part[i] % delta_t) / delta_t * 2 * math.pi




def trig_approx(f, t, K = 3):
  n = f.size
  
  A = np.zeros([n,2*K+1])
  for i in range(0,n):
    A[i,0] = 1 / 2
    for k in range(0,K):
      A[i,1+k] = math.cos((1 + k) * t[i])
      A[i,1+K+k] = math.sin((1 + k) * t[i])

  b = np.reshape(f, [n,1])
  return np.linalg.solve(A.T @ A, A.T @ b)

def eval_trig_approx(coefs, t_i):
  K = math.floor(coefs.size/2)

  a = np.zeros([K,1])
  b = np.zeros([K,1])

  for k in range(0,K):
    a[k] = coefs[1+k]
    b[k] = coefs[1+K+k]

  f_i = coefs[0]/2

  for k in range(0,K):
    f_i = f_i + a[k] * math.cos((1 + k) * t_i)
    f_i = f_i + b[k] * math.sin((1 + k) * t_i)

  return f_i


res = 100

def calc_trig_line(coefs, a, b, res = 100):
  f = np.zeros([res,1])
  i = 0
  for t in np.linspace(a, b, res):
    f[i] = eval_trig_approx(coefs, t)
    i = i + 1

  return f

coefsX = trig_approx(bmX_part, t_cutted, 1)
approxX = calc_trig_line(coefsX, 0, 2*math.pi, res)

coefsY = trig_approx(bmY_part, t_cutted, 1)
approxY = calc_trig_line(coefsY, 0, 2*math.pi, res)

roundabout = np.linspace(0, 2*math.pi, res)

fig, ax = plt.subplots()
ax.plot(t_cutted, bmX_part, '.')
ax.plot(t_cutted, bmY_part, '.')
ax.plot(roundabout, approxX)
ax.plot(roundabout, approxY)
st.pyplot(fig)


phiX_shift = st.slider('x shift', -math.pi, math.pi, 0.0, 0.01)
phiY_shift = st.slider('y shift', -math.pi, math.pi, 0.0, 0.01)

phiX = math.atan2(coefsX[1], coefsX[2])
if phiX < 0:
  phiX = phiX * -1
else:
  phiX = 2*math.pi - phiX

phiX = phiX - math.pi/2 + phiX_shift

# approxX_max_i = np.argmax(approxX)
# st.write(roundabout[approxX_max_i]-math.pi/2, phiX)

phiY = math.atan2(coefsY[1], coefsY[2])
if phiY < 0:
  phiY = phiY * -1
else:
  phiY = 2*math.pi - phiY + phiY_shift

approxX_shifted = calc_trig_line(coefsX, phiX, 2*math.pi + phiX, res)
approxY_shifted = calc_trig_line(coefsY, phiY, 2*math.pi + phiY, res)

tX_cutted = np.zeros((n_part,1))
tY_cutted = np.zeros((n_part,1))
for i in range(0,n_part):
  tX_cutted[i] = ((t_part[i] % delta_t) / delta_t * 2 * math.pi - phiX) % (2*math.pi)
  tY_cutted[i] = ((t_part[i] % delta_t) / delta_t * 2 * math.pi - phiY) % (2*math.pi)

fig, ax = plt.subplots()
ax.plot(tX_cutted, bmX_part, '.')
ax.plot(tY_cutted, bmY_part, '.')
ax.plot(roundabout, approxX_shifted)
ax.plot(roundabout, approxY_shifted)
st.pyplot(fig)


tX_sorted_ind = np.argsort(tX_cutted.flatten())
tY_sorted_ind = np.argsort(tY_cutted.flatten())

bmX_rearranged = np.take_along_axis(bmX_part.flatten(), tX_sorted_ind, axis=-1) 
bmY_rearranged = np.take_along_axis(bmY_part.flatten(), tY_sorted_ind, axis=-1) 

# tX_i_min = np.argmin(tX_cutted)
# tY_i_min = np.argmin(tY_cutted)

# bmX_rearranged = np.zeros((n_part,1))
# bmY_rearranged = np.zeros((n_part,1))
# for i in range(0,n_part):
#   bmX_rearranged[i] = bmX_part[(i+tX_i_min) % n_part]
#   bmY_rearranged[i] = bmY_part[(i+tY_i_min) % n_part]

fig, ax = plt.subplots()
ax.plot(bmX_part, bmY_part, '.')
ax.plot(bmX_rearranged, bmY_rearranged, 'x')
ax.axis('equal')
st.pyplot(fig)