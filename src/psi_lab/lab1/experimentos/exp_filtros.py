"""Experimentos reproducibles de las tareas 4.2 y 4.3: filtros FIR por ventana."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psi_lab.lab1.filtros import filtrar, fir_pasabajos, respuesta_frecuencia


def _cruce(freqs, amplitud, nivel):
    """
    entrada: freqs y amplitud, respuesta del filtro / nivel, amplitud buscada.
    descripcion: Interpola el primer cruce descendente del nivel indicado.
    salida: Frecuencia del cruce en Hz.
    """
    indices = np.flatnonzero((amplitud[:-1] >= nivel) & (amplitud[1:] < nivel))
    if indices.size == 0:
        raise ValueError(f"No se encontró un cruce descendente del nivel {nivel}.")
    i = indices[0]
    return freqs[i] + (nivel - amplitud[i]) * (freqs[i + 1] - freqs[i]) / (amplitud[i + 1] - amplitud[i])


def _guardar_tabla(out_dir, filas, explicacion):
    """
    entrada: out_dir, destino / filas, mediciones / explicacion, método empleado.
    descripcion: Exporta las mediciones y documenta cómo se obtuvieron.
    salida: Archivos mediciones.csv y resultados.md.
    """
    columnas = list(filas[0])
    with (out_dir / "mediciones.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    lineas = [explicacion, "", "| " + " | ".join(columnas) + " |", "| " + " | ".join(["---"] * len(columnas)) + " |"]
    for fila in filas:
        valores = [f"{valor:.6f}" if isinstance(valor, (float, np.floating)) else str(valor) for valor in fila.values()]
        lineas.append("| " + " | ".join(valores) + " |")
    (out_dir / "resultados.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _caracterizacion(out_dir):
    """
    entrada: out_dir, carpeta para los resultados de la Tarea 4.2.
    descripcion: Compara cuatro longitudes y tres ventanas con fc=100 y fs=1000 Hz.
    salida: Figura PNG/PDF, mediciones CSV y explicación con tabla Markdown.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filas = []
    fig, ejes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    ventanas = [("rect", "Rectangular"), ("hamming", "Hamming"), ("hann", "Hann")]
    try:
        for eje, (ventana, nombre) in zip(ejes, ventanas):
            for n_coef in [21, 51, 101, 201]:
                h = fir_pasabajos(100, 1000, n_coef, ventana)
                freqs, H = respuesta_frecuencia(h, 1000, n=32769)
                amplitud = np.abs(H)
                f90 = _cruce(freqs, amplitud, 0.9)
                f10 = _cruce(freqs, amplitud, 0.1)
                # Excluir la transición antes de medir los lóbulos de rechazo.
                minimos = np.flatnonzero((amplitud[1:-1] < amplitud[:-2]) & (amplitud[1:-1] <= amplitud[2:])) + 1
                candidatos = minimos[freqs[minimos] > max(100, f10)]
                if candidatos.size == 0:
                    raise ValueError("No se encontró el primer mínimo de rechazo.")
                inicio = candidatos[0]
                rechazo_db = 20 * np.log10(np.max(amplitud[inicio:]))
                filas.append({
                    "ventana": nombre,
                    "n_coef": n_coef,
                    "f_90_Hz": float(f90),
                    "f_10_Hz": float(f10),
                    "transicion_Hz": float(f10 - f90),
                    "inicio_rechazo_Hz": float(freqs[inicio]),
                    "max_rechazo_dB": float(rechazo_db),
                    "atenuacion_dB": float(-rechazo_db),
                })
                # El piso se usa solo al dibujar, no al medir.
                eje.plot(freqs, 20 * np.log10(np.maximum(amplitud, 1e-8)), label=f"N = {n_coef}", linewidth=1)
            eje.axvline(100, color="black", linestyle="--", label="Corte: 100 Hz")
            eje.axhline(-40, color="gray", linestyle=":", label="−40 dB")
            eje.set(title=nombre, ylabel="20 log₁₀ |H(f)| [dB]", xlim=(0, 500), ylim=(-100, 3))
            eje.grid(alpha=0.25)
            eje.legend(loc="upper right", ncol=2, fontsize=9)
        ejes[-1].set_xlabel("Frecuencia [Hz]")
        fig.suptitle("Tarea 4.2 · Pasabajos por ventana · fc = 100 Hz, fs = 1000 Hz")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / "respuestas_frecuencia.png", dpi=180)
        fig.savefig(out_dir / "respuestas_frecuencia.pdf")
    finally:
        plt.close(fig)
    _guardar_tabla(out_dir, filas, (
        "# Tarea 4.2: caracterización de filtros FIR\n\n"
        "fc = 100 Hz, fs = 1000 Hz. Respuesta directa en 32769 puntos entre 0 y 500 Hz "
        "(paso aproximado de 0,01526 Hz).\n\n"
        "Transición: distancia entre los primeros cruces descendentes de |H| = 0,9 y 0,1, "
        "interpolados linealmente (aproximadamente −0,92 y −20 dB).\n\n"
        "Rechazo: máximo de 20 log10|H| desde el primer mínimo local posterior al corte "
        "y al cruce de 0,1 hasta Nyquist. Se informa el comienzo de cada intervalo: "
        "las bandas medidas son distintas y excluyen la transición. La atenuación es "
        "el negativo de ese máximo. Los extremos se estiman sobre una malla finita.\n\n"
        "El gráfico usa un piso de −160 dB y muestra hasta −100 dB; las mediciones no se "
        "recortan. Más coeficientes estrechan la transición; la rectangular presenta "
        "una transición más estrecha y lóbulos mayores que Hann y Hamming en estos diseños."
    ))


def _chirp(out_dir):
    """
    entrada: out_dir, carpeta para los resultados de la Tarea 4.3.
    descripcion: Filtra un chirp de 0 a 500 Hz con Hamming de 101 coeficientes.
    salida: Figuras temporales, mediciones RMS en CSV y explicación Markdown.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fs, duracion = 1000, 5
    t = np.arange(fs * duracion) / fs
    # f(t)=100*t Hz: integrar la frecuencia da la fase 2*pi*50*t².
    x = np.sin(2 * np.pi * 50 * t**2)
    h = fir_pasabajos(100, fs, 101)
    y = filtrar(x, h)
    # RMS centrado de 101 muestras, con sumas acumuladas sin np.convolve.
    tiempo_rms = t[50:-50]
    valores_rms = []
    for senal in [x, y]:
        acumulada = np.concatenate(([0.0], np.cumsum(senal**2)))
        valores_rms.append(np.sqrt(np.maximum((acumulada[101:] - acumulada[:-101]) / 101, 0)))
    cambio_db = 20 * np.log10(np.maximum(valores_rms[1], 1e-12) / np.maximum(valores_rms[0], 1e-12))
    # Se excluyen ventanas RMS que contienen muestras afectadas por los bordes.
    interior = (tiempo_rms >= 0.1) & (tiempo_rms < duracion - 0.1)
    fig, ejes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    try:
        ejes[0].plot(t, x, color="gray", linewidth=0.65)
        ejes[0].set_title("Entrada: chirp lineal de 0 a 500 Hz en 5 s")
        ejes[1].plot(t, y, linewidth=0.8)
        ejes[1].set_title("Salida: pasabajos Hamming, N = 101, fc = 100 Hz")
        for eje in ejes[:2]:
            eje.set(ylabel="Amplitud", ylim=(-1.15, 1.15))
        ejes[2].plot(tiempo_rms[interior], cambio_db[interior], color="tab:purple")
        ejes[2].axhline(-40, color="gray", linestyle=":", label="−40 dB")
        ejes[2].set(title="Cambio local de RMS · ventana de 101 muestras", ylabel="Cambio de RMS [dB]", ylim=(-90, 3))
        ejes[2].set_xlabel("Tiempo [s] · frecuencia instantánea f(t) = 100 t Hz")
        for eje in ejes:
            eje.axvline(1, color="tab:red", linestyle="--", label="100 Hz en t = 1 s")
            eje.axvspan(0, 0.05, color="orange", alpha=0.15)
            eje.axvspan(duracion - 0.05, duracion, color="orange", alpha=0.15)
            eje.set_xlim(0, duracion)
            eje.grid(alpha=0.25)
            eje.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "chirp_filtrado.png", dpi=180)
        fig.savefig(out_dir / "chirp_filtrado.pdf")
    finally:
        plt.close(fig)

    fig, eje = plt.subplots(figsize=(11, 4))
    try:
        zoom = (t >= 0.85) & (t <= 1.15)
        eje.plot(t[zoom], x[zoom], color="gray", alpha=0.7, label="Entrada")
        eje.plot(t[zoom], y[zoom], label="Salida con retardo compensado")
        eje.axvline(1, color="tab:red", linestyle="--", label="100 Hz")
        eje.set(xlabel="Tiempo [s]", ylabel="Amplitud", title="Detalle temporal alrededor del corte")
        eje.legend()
        eje.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "detalle_corte.png", dpi=180)
    finally:
        plt.close(fig)
    filas = []
    for nombre, inicio, fin in [("Paso", 0.2, 0.8), ("Alrededor del corte", 0.95, 1.05), ("Rechazo", 1.5, 4.5)]:
        region = (t >= inicio) & (t < fin)
        entrada_rms = np.sqrt(np.mean(x[region]**2))
        salida_rms = np.sqrt(np.mean(y[region]**2))
        filas.append({
            "region": nombre,
            "inicio_s": inicio,
            "fin_s_excluido": fin,
            "frecuencia_inicio_Hz": 100 * inicio,
            "frecuencia_fin_Hz_excluida": 100 * fin,
            "rms_entrada": float(entrada_rms),
            "rms_salida": float(salida_rms),
            "cambio_rms_dB": float(20 * np.log10(salida_rms / entrada_rms)),
        })
    _guardar_tabla(out_dir, filas, (
        "# Tarea 4.3: filtrado de un chirp\n\n"
        "fs = 1000 Hz, duración nominal de 5 s, 5000 muestras en [0, 5) s. "
        "x(t) = sin(2*pi*50*t²), con frecuencia instantánea f(t) = 100*t Hz. "
        "La última muestra corresponde a 499,9 Hz; 500 Hz es el extremo nominal no incluido.\n\n"
        "Se aplica nuestro pasabajos Hamming de 101 coeficientes y corte 100 Hz. "
        "El chirp cruza el corte en t = 1 s; la atenuación aparece gradualmente. "
        "El retardo de 50 muestras se compensa fuera de línea. Los primeros y últimos "
        "0,05 s pueden estar afectados por la extensión con ceros y se marcan en naranja.\n\n"
        "El tercer panel muestra 20 log10(RMS salida / RMS entrada) en ventanas "
        "centradas de 101 muestras, excluyendo las que tocan los bordes afectados. "
        "Es una medida temporal suavizada, no una respuesta en frecuencia exacta.\n\n"
        "La tabla calcula RMS sobre cada intervalo completo, con extremo final excluido. "
        "No promedia dB del gráfico. Su reducción de RMS en rechazo no equivale al peor "
        "rechazo para un tono medido en la Tarea 4.2."
    ))


def generar(out_dir="out"):
    """
    entrada: out_dir, directorio de salida ("out" por defecto).
    descripcion: Genera las figuras y tablas FIR sin depender de archivos temporales.
    salida: Resultados en las subcarpetas tarea4_2 y tarea4_3 de out_dir.
    """
    out_dir = Path(out_dir)
    _caracterizacion(out_dir / "tarea4_2")
    _chirp(out_dir / "tarea4_3")
    print(f"exp_filtros: resultados guardados en {out_dir}")


if __name__ == "__main__":
    generar()
