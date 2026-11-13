import numpy as np

from psi_lab.lab1.dft import dft, espectro, idft

rng = np.random.default_rng(0)
# Test para comprobar dft.py

#1. test de coincidencia de resultados
def test_dft_coincidencia_fft():
    # Entrada de prueba
    x = rng.standard_normal(20)
    # Resultado esperado usando la función dft de la librería numpy
    expected_output = np.fft.fft(x)
    # Resultado obtenido usando la función dft implementada
    output = dft(x)
    # Comparar los resultados
    assert np.allclose(output, expected_output)
    # Si sale bien debe aparecer PASSED

#2. test de simetria conjugada para una senial real X[N-k] = conj(X[k])
def test_dft_simetria_conjugada():
    # Entrada
    x = rng.standard_normal(20)
    X = dft(x)
    N = len(x)
    for k in range(1, N):
        assert np.allclose(X[N-k], np.conj(X[k]))

#3. test de Parseval: la energia del dominio del tiempo es igual a la energia en el dominio de la frecuencia
def test_dft_parseval():
    # Entrada
    x = rng.standard_normal(20)
    X = dft(x)
    # Energia en el dominio del tiempo
    energy_time = np.sum(np.abs(x)**2)
    # Energia en el dominio de la frecuencia
    energy_freq = np.sum(np.abs(X)**2) / len(X)
    assert np.allclose(energy_time, energy_freq)

#4. test de idft: la transformada inversa debe recuperar la señal original
def test_idft_recuperacion():
    x = rng.standard_normal(20)
    X = dft(x) # Aqui se hace la transformada directa
    x_recovered = idft(X) # Aqui se hace la transformada inversa
    assert np.allclose(x, x_recovered) # Se verifican que sean iguales la señal original y la recuperada

def test_espectro():
    fs = 100
    t = np.arange(1000) / fs
    x = 3 * np.sin(2* np.pi * 10 * t) # Señal con 3 de amplitud y seno de 10 Hz
    for v in ["rect", "hann", "hamming"]:
        f, a = espectro(x, fs, ventana=v)
        i = np.argmax(a)
        assert np.isclose(f[i], 10)
        assert np.isclose(a[i], 3)

    
