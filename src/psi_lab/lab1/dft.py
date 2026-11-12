import numpy as np


def dft(x):
    """ DFT por definicion: O(n^2): X[k] = sum_n X[n] exp (-2j pi k n / N).
    x: array de entrada
    N: cantidad de muestras o largo array
    x[n]: valor de la muestra n
    k: indice de la frecuencia
    X[k]: cuanto de la frecuencia k hay en la señal, valor que se guarda"""
    N = len(x)
    X = np.zeros(N, dtype=complex) # Array de ceros para almacenar los resultados de la DFT
    for k in range (N):
        for n in range(N):
            X[k] += x[n]* np.exp(-2j * np.pi * k * n / N)
    return X


def idft(X):
    """ DFT inversa por definicion: O(n^2): x[n] = (1/N) sum_k X[k] exp (2j pi k n / N)."""
    N = len(X)
    x = np.zeros(N, dtype=complex) # Array de ceros para almacenar los resultados de la IDFT
    for n in range (N):
        for k in range(N):
            x[n] += X[k]* np.exp(2j * np.pi * k * n / N)
        x[n] /= N
    return x