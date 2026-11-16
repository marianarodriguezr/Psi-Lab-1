"""Diseño de filtros FIR mediante sinc truncada y ventanas simétricas."""

from numbers import Integral, Real

import numpy as np


def fir_pasabajos(fc, fs, n_coef, ventana="hamming"):
    """
    entrada: fc, frecuencia de corte real y finita en Hz, con 0 < fc < fs / 2 /
        fs, frecuencia de muestreo real, finita y positiva en Hz /
        n_coef, número entero positivo impar de coeficientes /
        ventana, "rect", "hann" o "hamming" (predeterminada).
    descripcion: Diseña un FIR pasabajos multiplicando la sinc ideal por una
        ventana y normalizando por la suma para obtener ganancia uno en continua.
        La longitud y la ventana determinan la transición y la atenuación.
        El retardo de grupo es (n_coef - 1) / 2 muestras.
        Lanza ValueError si alguna entrada no es válida.
    salida: Vector NumPy de n_coef coeficientes reales y simétricos.
    """
    # Se excluyen booleanos explícitamente porque bool es un subtipo de int.
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, Real) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser un número real finito y positivo.")
    if isinstance(fc, (bool, np.bool_)) or not isinstance(fc, Real) or not np.isfinite(fc) or not 0 < fc < fs / 2:
        raise ValueError("fc debe ser real y finita, con 0 < fc < fs / 2.")
    # La longitud impar permite un centro y un retardo de grupo enteros.
    if isinstance(n_coef, (bool, np.bool_)) or not isinstance(n_coef, Integral) or n_coef <= 0 or n_coef % 2 == 0:
        raise ValueError("n_coef debe ser un entero positivo impar.")

    # Hann y Hamming suavizan los extremos para reducir los lóbulos laterales,
    # a costa de una transición más ancha que con la ventana rectangular.
    if ventana == "rect":
        w = np.ones(n_coef)
    elif ventana == "hann":
        w = np.hanning(n_coef)
    elif ventana == "hamming":
        w = np.hamming(n_coef)
    else:
        raise ValueError("ventana debe ser 'rect', 'hann' o 'hamming'.")

    # Centrar los índices alrededor de cero hace simétricos los coeficientes.
    k = np.arange(n_coef) - (n_coef - 1) / 2
    # El corte se expresa como fracción de Nyquist: fc / (fs / 2).
    corte_normalizado = 2 * (fc / fs)
    # np.sinc(u) = sin(pi*u)/(pi*u) y resuelve el valor en u = 0 como 1.
    h = corte_normalizado * np.sinc(corte_normalizado * k) * w
    # H(0) es la suma de los coeficientes: al dividir por ella queda en uno.
    return h / np.sum(h)


def fir_pasaaltos(fc, fs, n_coef, ventana="hamming"):
    """
    entrada: fc, frecuencia de corte real y finita en Hz, con 0 < fc < fs / 2 /
        fs, frecuencia de muestreo real, finita y positiva en Hz /
        n_coef, número entero positivo impar de coeficientes /
        ventana, "rect", "hann" o "hamming" (predeterminada).
    descripcion: Diseña un FIR pasaaltos por inversión espectral del pasabajos.
        Resta sus coeficientes de un impulso situado en el centro, cancelando
        la continua y conservando el retardo de (n_coef - 1) / 2 muestras.
        La longitud y la ventana determinan la transición y la atenuación.
        Con un solo coeficiente devuelve el filtro nulo, sin banda de paso.
        Lanza ValueError si alguna entrada no es válida.
    salida: Vector NumPy de n_coef coeficientes reales y simétricos.
    """
    # Reutilizar el pasabajos conserva sus validaciones y su ganancia en continua.
    h = -fir_pasabajos(fc, fs, n_coef, ventana)
    # El impulso debe tener el mismo retardo que el pasabajos para cancelarlo.
    h[(n_coef - 1) // 2] += 1
    return h


def fir_pasabanda(f1, f2, fs, n_coef, ventana="hamming"):
    """
    entrada: f1, frecuencia de corte inferior real y finita en Hz /
        f2, frecuencia de corte superior real y finita en Hz,
        con 0 < f1 < f2 < fs / 2 /
        fs, frecuencia de muestreo real, finita y positiva en Hz /
        n_coef, número entero positivo impar de coeficientes /
        ventana, "rect", "hann" o "hamming" (predeterminada).
    descripcion: Diseña un FIR pasabanda restando el pasabajos de corte f1
        al de corte f2. Conserva el retardo de (n_coef - 1) / 2 muestras.
        La longitud y la ventana determinan la transición y la atenuación;
        una banda demasiado estrecha puede no alcanzar ganancia próxima a uno.
        Con un solo coeficiente devuelve el filtro nulo, sin banda de paso.
        Lanza ValueError si alguna entrada no es válida.
    salida: Vector NumPy de n_coef coeficientes reales y simétricos.
    """
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, Real) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser un número real finito y positivo.")
    if isinstance(f1, (bool, np.bool_)) or not isinstance(f1, Real) or not np.isfinite(f1) or not 0 < f1 < fs / 2:
        raise ValueError("f1 debe ser real y finita, con 0 < f1 < fs / 2.")
    if isinstance(f2, (bool, np.bool_)) or not isinstance(f2, Real) or not np.isfinite(f2) or not f1 < f2 < fs / 2:
        raise ValueError("f2 debe ser real y finita, con f1 < f2 < fs / 2.")

    # Ambos pasabajos comparten longitud, ventana y retardo para poder restarlos.
    superior = fir_pasabajos(f2, fs, n_coef, ventana)
    inferior = fir_pasabajos(f1, fs, n_coef, ventana)
    # Sus ganancias unitarias en continua se cancelan; queda la banda intermedia.
    return superior - inferior


def fir_rechazabanda(f1, f2, fs, n_coef, ventana="hamming"):
    """
    entrada: f1, frecuencia de corte inferior real y finita en Hz /
        f2, frecuencia de corte superior real y finita en Hz,
        con 0 < f1 < f2 < fs / 2 /
        fs, frecuencia de muestreo real, finita y positiva en Hz /
        n_coef, número entero positivo impar de coeficientes /
        ventana, "rect", "hann" o "hamming" (predeterminada).
    descripcion: Diseña un FIR rechazabanda por inversión espectral del
        pasabanda. Atenúa la banda entre f1 y f2 y conserva las exteriores,
        con un retardo de (n_coef - 1) / 2 muestras y ganancia uno en continua.
        La longitud y la ventana determinan la transición y la atenuación;
        una banda demasiado estrecha puede no alcanzar el rechazo deseado.
        Con un solo coeficiente devuelve [1], sin atenuar ninguna frecuencia.
        Lanza ValueError si alguna entrada no es válida.
    salida: Vector NumPy de n_coef coeficientes reales y simétricos.
    """
    # El pasabanda valida las entradas y aporta los coeficientes de la banda.
    h = -fir_pasabanda(f1, f2, fs, n_coef, ventana)
    # El impulso central comparte el retardo del pasabanda: su resta lo invierte.
    h[(n_coef - 1) // 2] += 1
    return h


def respuesta_frecuencia(h, fs, n=4096):
    """
    entrada: h, vector no vacío de coeficientes numéricos finitos, reales o complejos /
        fs, frecuencia de muestreo real, finita y positiva en Hz /
        n, número entero positivo de puntos (4096 por defecto).
    descripcion: Evalúa la respuesta en frecuencia mediante la suma directa
        H(f) = suma de h[k] * exp(-j * 2 * pi * f * k / fs), sin utilizar FFT.
        Usa n frecuencias equiespaciadas entre 0 y fs / 2, incluidos los extremos.
        Con n = 1 evalúa únicamente la continua. Admite longitudes pares e impares.
        El costo es O(n * len(h)) y la memoria auxiliar es O(n).
        Lanza ValueError si alguna entrada no es válida.
    salida: Tupla (freqs, H), con dos vectores de n elementos: frecuencias en Hz
        y respuesta compleja, que conserva información de amplitud y fase.
    """
    h = np.asarray(h)
    if h.ndim != 1 or h.size == 0 or not np.issubdtype(h.dtype, np.number) or not np.all(np.isfinite(h)):
        raise ValueError("h debe ser un vector no vacío de coeficientes numéricos finitos.")
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, Real) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser un número real finito y positivo.")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("n debe ser un entero positivo.")

    freqs = np.linspace(0, fs / 2, n)
    # La frecuencia angular recorre de 0 a pi radianes por muestra.
    omega = np.linspace(0, np.pi, n)
    H = np.zeros(n, dtype=complex)
    # Acumular por coeficiente evita construir una matriz de n por len(h).
    for k, coeficiente in enumerate(h):
        H += coeficiente * np.exp(-1j * omega * k)
    return freqs, H


def filtrar(x, h):
    """
    entrada: x, vector no vacío de muestras numéricas finitas, reales o complejas /
        h, vector no vacío de coeficientes reales finitos, simétrico y de longitud impar.
    descripcion: Calcula la convolución lineal por suma directa de aportes
        desplazados y compensa el retardo de grupo
        (len(h) - 1) / 2 recortando la salida desde ese índice.
        Supone ceros fuera de x, por lo que pueden aparecer efectos de borde.
        Es un procesamiento fuera de línea: la compensación requiere muestras futuras.
        Lanza ValueError si alguna entrada no es válida.
    salida: Vector NumPy de len(x) muestras filtradas, reales o complejas según x.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    if x.ndim != 1 or x.size == 0 or not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError("x debe ser un vector no vacío de muestras numéricas finitas.")
    if h.ndim != 1 or h.size == 0 or not np.issubdtype(h.dtype, np.number) or np.iscomplexobj(h) or not np.all(np.isfinite(h)):
        raise ValueError("h debe ser un vector no vacío de coeficientes reales finitos.")
    if h.size % 2 == 0 or not np.allclose(h, h[::-1], rtol=1e-10, atol=1e-12):
        raise ValueError("h debe ser simétrico y de longitud impar para compensar su retardo.")

    # Promover a punto flotante evita desbordamientos al convolucionar enteros.
    tipo = np.result_type(x.dtype, h.dtype, np.float64)
    x = x.astype(tipo, copy=False)
    h = h.astype(tipo, copy=False)
    completa = np.zeros(x.size + h.size - 1, dtype=tipo)
    # Por definición, cada h[k] aporta h[k] * x[m - k] a la salida.
    # El segmento que comienza en k desplaza la señal sin envolver sus extremos.
    for k, coeficiente in enumerate(h):
        completa[k:k + x.size] += coeficiente * x
    retardo = (h.size - 1) // 2
    # El recorte conserva len(x) incluso cuando el filtro es más largo que la señal.
    return completa[retardo:retardo + x.size]
