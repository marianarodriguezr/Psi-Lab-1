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


def _ventana(nombre, N):
    """Retorna la ventana 'rect' 'hann' o 'hamming' de largo N."""
    n = np.arange(N)
    w = np.zeros(N)
    if nombre == "rect":
        return np.ones(N) # Retorna una ventana rectangular, arreglo con valores 1
    elif nombre =='hann':
        for n in range(N):
            w[n] = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) # Calcula la ventana de Hann
        return w
    elif nombre == 'hamming':
        for n in range(N):
            w[n] = 0.54 -0.46*np.cos(2*np.pi*n/(N-1)) # Calcula la ventana de Hamming
        return w
    raise ValueError(f"Ventana desconocida: {nombre}") # Lanza un error si la ventana no es reconocida

    
def espectro(x, fs, ventana="rect", nfft=None):
    """Espectro de amplitud de un lado, ventana en {"rect", "hann", "hamming"}; 
    nfft < len(x) rellena con ceros. Retorna (freqs, amp)"""
    """
    rect = w[n] = 1 no cambia nada
    hann = w[n] = 0.5 - 0.5 cos(2 pi n / (N-1))
    hamming = w[n] = 0.54 - 0.46 cos(2 pi n / (N-1))
    """
    x = np.asarray(x, dtype=float) # Se convierte la entrada en arreglo
    N = len(x)
    if nfft is None:
        nfft = N
    w = _ventana(ventana, N)
    XW = x * w # Se multiplica la señal por la ventana
    z = np.zeros(nfft)
    for i in range(N):
        z[i] = XW[i] # Se copia la señal enventanada en los primeros N lugares
    X = dft(z) # Se calcula la DFT del arreglo rellenado con ceros y enventanada
    K = nfft//2 + 1
    amp = np.zeros(K)
    for i in range(K):
        amp[i] = np.abs(X[i])
        amp[i] /= np.sum(w)
        if i != 0:
            amp[i] *= 2
        if i == K-1 and nfft % 2 == 0:
            amp[i] /= 2
    
    freq = np.arange(K) * fs / nfft # Se crea arreglo de decimales 
    return freq, amp
    

    
