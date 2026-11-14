"""Experimentos de la parte comun para dft, resolucion y fuga espectral"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from psi_lab.lab1.dft import dft, espectro


def _tiempos(out_dir):
    """ Compara los tiempos de dft y np.fft.fft para N de distintos tamanos"""

    # Se calculan los tiempos
    dft_times = []
    fft_times = []
    tamanosN = []
    
    np.fft.fft(np.random.randn(16))  # Preparacion para la biblioteca Numpy para calcular y reservar memoria

    # Bucle para valores exponenciales
    for i in range(4, 13):
        x = np.random.randn(2**i)
        inicio_dft = time.perf_counter()
        dft(x)
        fin_dft = time.perf_counter()
        tiempo_total_dft = fin_dft - inicio_dft
        dft_times.append(tiempo_total_dft)

        # Como fft es tan rapida para las mediciones, se mide el calculo del tiempo 1000 veces
        # sacando el promedio de los tiempos para cada N, siendo un valor mucho mas confiable
        reps = 1000
        inicio_fft = time.perf_counter()
        for _ in range(reps):
            np.fft.fft(x)
        fin_fft = time.perf_counter()
        fft_times.append((fin_fft - inicio_fft) / reps)

        tamanosN.append(2**i)

    # Calculo de las pendientes
    pendiente_dft = np.polyfit(np.log(tamanosN), np.log(dft_times), 1)[0]
    
    # Como la curva de fft es plana, se calcula la pendiente desde los valores
    # donde la curva crece (los 4 N mas grandes, N>=512)
    pendiente_fft = np.polyfit(np.log(tamanosN[-4:]), np.log(fft_times[-4:]), 1)[0]
    print(f"Pendiente dft: {pendiente_dft:.2f}")
    print(f"Pendiente fft: {pendiente_fft:.2f}")

    # Se grafica en escala log-log
    plt.figure(figsize=(10, 10))
    plt.loglog(tamanosN, dft_times, "o-", label=f"dft con pendiente={pendiente_dft:.2f}")
    plt.loglog(tamanosN, fft_times, "o-", label=f"np.fft.fft con pendiente={pendiente_fft:.2f}")
    plt.xlabel("Eje X = tamaños de N")
    plt.ylabel("Eje Y = tiempos[s]")
    plt.title("Gráfico de comparacion de tiempos para dft y np.fft.fft en escala log-log")
    plt.legend()
    plt.grid(True)

    # Se guarda el grafico
    plt.savefig(os.path.join(out_dir, "dft_fft_comparacion_loglog.png"))

    plt.close()

# Tarea 2: punto 2:
# Resolucion
def _resolucion(out_dir):
    """ Dos tonos de 50 y 52 Hz, fs = 1000 Hz, duraciones T e {0.25; 0.5; 1; 2} s"""
    # Se crea el arreglo con las seniales
    frecuencias = []
    amplitudes = []
    fs = 1000
    T = [0.25, 0.5, 1, 2]
    for Ti in T:
        N = int(Ti * fs)
        t = np.arange(N)/fs # Arreglos de tiempos
        x = np.sin(2* np.pi * 50 * t) + np.sin(2* np.pi * 52 *t)
         # Se llama a la funcion espectro con fs=1000 y el arreglo de seniales x
        freq, amp = espectro(x, fs)
        frecuencias.append(freq)
        amplitudes.append(amp)

    # Se genera el grafico
    fig, graf = plt.subplots(4, 1, figsize=(10, 10))
    for i in range(4):
        graf[i].plot(frecuencias[i], amplitudes[i], "o-")
        graf[i].set_xlim(40, 65)
        graf[i].set_xlabel("Frecuencia (Hz)")
        graf[i].set_ylabel("Amplitud")
        graf[i].set_title(f"Grafico para T={T[i]} s, Δf={1/T[i]:.2f} Hz")

    fig.suptitle("Resolucion: tonos de 50 y 52 Hz con distintas duraciones")
    fig.tight_layout() # Para que no se encimen los graficos
    fig.savefig(os.path.join(out_dir, "resolucion_dft_graficos.png"))
    plt.close(fig)

# Tarea 2: punto 3
# Relleno con ceros
def _relleno(out_dir):
    """Rellenos con ceros: T = 0.25 s con nfft = N y nfft = 8N"""
    T = 0.25
    fs = 1000
    N = int(T*fs)
    t = np.arange(N)/fs
    x = np.sin(2*np.pi * 50* t) + np.sin(2*np.pi * 52 * t)
    freq1, amp1 = espectro(x, fs)
    freq2, amp2 = espectro(x, fs, nfft=8*N)
    fig, graf = plt.subplots(figsize=(10, 5))
    graf.plot(freq2, amp2, "-", label=f"T={T}s sin nfft=8N")
    graf.plot(freq1, amp1, "o", markersize=5, label=f"T={T}s con nfft=N")
    graf.set_xlim(30, 72)
    graf.set_xlabel("Frecuencia (Hz)")
    graf.set_ylabel("Amplitud")
    graf.axvline(50, color="gray", linestyle="--")
    graf.axvline(52, color="gray", linestyle="--")
    fig.suptitle(f"Relleno con ceros: T={T}s y nfft=N vs nfft = 8N")
    graf.legend()
    graf.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "relleno_con_ceros_dft_grafico.png"))
    plt.close(fig)

def _fuga(out_dir):
    """Fuga espectral: tono de 50 Hz de amplitud 1 y otro de 80 Hz de amplitud 10^-3
       Comparacion rectangular y Hann en dB"""
    fs = 1000
    N = 1024 # Recomendacion para visualizar la fuga
    t = np.arange(N)/fs
    x = np.sin(2*np.pi * 50 * t) + 1e-3 * np.sin(2*np.pi * 80 * t)
    ventanas = ["rect", "hann"]
    fig, graf = plt.subplots(len(ventanas), 1, figsize=(10, 10))

    for i, v in enumerate(ventanas):
        f, a = espectro(x, fs, ventana=v)
        a_db = 20 * np.log10(np.maximum(a, 1e-12)/1.0) # Se pasa la amplitud a db
        graf[i].plot(f, a_db, "o-")
        graf[i].set_xlim(0, 150)
        graf[i].set_ylim(-120, 5)
        graf[i].axvline(80, color="gray", linestyle="--") # Donde se deberia encontrar el tono debil
        graf[i].axhline(-60, color="gray", linestyle="--") # altura esperada 
        graf[i].set_xlabel("Frecuencia (Hz)")
        graf[i].set_ylabel("Amplitud (dB)")
        graf[i].set_title(f"Grafico de fuga con ventana={v}")

    fig.suptitle("Fuga y ventanas: Tonos de 50 y 80 Hz de amplitudes 1 y 1e-3 respectivamente con distintas ventanas")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fuga_ventanas_dft_graficos.png"))
    plt.close(fig)

# out_dir carpeta donde se van a guardar las figuras
# Esta es la misma funcion para todos los exp
def generar(out_dir="out"): 
    """Generar las figuras y tablas de esta seccion en out_dir"""
    os.makedirs(out_dir, exist_ok=True)
    _tiempos(out_dir)
    _resolucion(out_dir)
    _relleno(out_dir)
    _fuga(out_dir)
    

# Ejecuta el archivo solo
if __name__ == "__main__":
    generar() #Utiliza la carpeta out
