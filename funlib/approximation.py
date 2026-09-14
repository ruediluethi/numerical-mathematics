import numpy as np

import streamlit as st
import math

@st.cache_data
def trig_approx(f: np.ndarray, t: np.ndarray, K: int = 20) -> np.ndarray:
    """
    Approximates a given function using trigonometric series.

    Args:
        f (np.ndarray): The function values.
        t (np.ndarray): The 'time' values normalized as radians (0 to 2pi).
        K (int, optional): The order of the approximation. Defaults to 20.

    Returns:
        np.ndarray: The coefficients of the trigonometric series as complex array. first element is a0/2, then a1, i*b1, a2, i*b2, ...
    """
    n = len(f)
    k = np.arange(1, K + 1)
    A = np.zeros([n, 2*K + 1])
    A[:, 0] = 1/2
    A[:, 1:K+1] = np.cos(k * t[:, None])
    A[:, K+1:] = np.sin(k * t[:, None])
    b = np.reshape(f, [n, 1])
    coeffs = np.linalg.solve(A.T @ A, A.T @ b).flatten()
    # store the coefficients in a complex array. first element is a0/2, then a1, i*b1, a2, i*b2, ...
    a = coeffs[0:K+1]
    b = coeffs[K+1:2*K+1]
    b = np.insert(b, 0, 0)
    return a + 1j*b

def eval_trig_approx(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Evaluates the trigonometric approximation at given time values.

    Args:
        t (np.ndarray): The 'time' values normalized as radians (0 to 2pi).
        a (np.ndarray): The coefficients of the trigonometric series as a complex array.
        l (int): The number of terms in the series.

    Returns:
        np.ndarray: The approximated function values.
    """
    l = len(a)
    k = np.arange(1, l)
    f = 0.5 * np.real(a[0]) + np.sum(np.real(a[1:]) * np.cos(k*t[:, None]) + np.imag(a[1:]) * np.sin(k*t[:, None]), axis=1)
    return f

def trig_integral(a: np.ndarray, start: float, end: float):
    """
    Calculates the integral from the trigonometric function defined throw the coefs from start to enf.
    Args:
        coefs (np.ndarray): The coefficients of the trigonometric series as a complex array.
        start (float): Integral starts here
        end (float): Integral ends here

    Returns:
        float: int_start^end 1/2 a_0 + sum_k=1^n Re(a_k)*cos(k*t) + Im(a_k)*sin(k*t) dt
    """
    l = len(a)
    k = np.arange(1, l)
    A = 0.5 * np.real(a[0]) * start + np.sum( np.real(a[1:]) * np.sin(k*start) / k) - np.sum( np.imag(a[1:]) * np.cos(k*start) / k)
    B = 0.5 * np.real(a[0]) * end   + np.sum( np.real(a[1:]) * np.sin(k*end  ) / k) - np.sum( np.imag(a[1:]) * np.cos(k*end  ) / k)
    
    return B - A
