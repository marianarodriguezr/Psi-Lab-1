"""Experimentos de la parte comun para dft, resolucion y fuga espectral"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from psi_lab.lab1.dft import dft


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


# out_dir carpeta donde se van a guardar las figuras
# Esta es la misma funcion para todos los exp
def generar(out_dir="out"): 
    """Generar las figuras y tablas de esta seccion en out_dir"""
    os.makedirs(out_dir, exist_ok=True)
    _tiempos(out_dir) 

# Ejecuta el archivo solo
if __name__ == "__main__":
    generar() #Utiliza la carpeta out
