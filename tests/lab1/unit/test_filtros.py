"""Validación de filtros FIR con referencias independientes de SciPy."""

import csv

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import convolve, firwin, freqz

from psi_lab.lab1.filtros import (
    filtrar,
    fir_pasaaltos,
    fir_pasabajos,
    fir_pasabanda,
    fir_rechazabanda,
    respuesta_frecuencia,
)


@pytest.mark.parametrize(
    "ventana, referencia",
    [("rect", "boxcar"), ("hann", "hann"), ("hamming", "hamming")],
)
def test_pasabajos_coincide_con_scipy(ventana, referencia):
    h = fir_pasabajos(1000, 8000, 101, ventana)
    esperado = firwin(101, 1000, fs=8000, window=referencia)

    assert h.shape == (101,)
    assert np.isrealobj(h)
    assert np.all(np.isfinite(h))
    assert_allclose(h, h[::-1], atol=1e-15)
    assert_allclose(np.sum(h), 1, atol=1e-14)
    assert_allclose(h, esperado, rtol=1e-12, atol=1e-15)


def test_pasabajos_hamming_predeterminada():
    assert_allclose(
        fir_pasabajos(1000, 8000, 101),
        fir_pasabajos(1000, 8000, 101, "hamming"),
    )


def test_pasabajos_ganancia_y_atenuacion():
    h = fir_pasabajos(1000, 8000, 101)
    freqs, respuesta = freqz(h, worN=16384, fs=8000)
    amplitud = np.abs(respuesta)

    # Se excluye la transición alrededor del corte de 1000 Hz.
    assert_allclose(amplitud[freqs <= 700], 1, atol=0.01, rtol=0)
    # |H| < 0.01 equivale a 20 log10(|H|) < -40 dB.
    assert np.max(amplitud[freqs >= 1300]) < 0.01


@pytest.mark.parametrize("ventana", ["rect", "hann", "hamming"])
def test_pasabajos_un_coeficiente(ventana):
    assert_allclose(fir_pasabajos(1000, 8000, 1, ventana), [1.0])


@pytest.mark.parametrize("fs", [0, -8000, np.nan, np.inf, True, "8000"])
def test_pasabajos_rechaza_fs_invalida(fs):
    with pytest.raises(ValueError, match="fs"):
        fir_pasabajos(1000, fs, 101)


@pytest.mark.parametrize("fc", [0, -1, 4000, 5000, np.nan, np.inf, True])
def test_pasabajos_rechaza_corte_invalido(fc):
    with pytest.raises(ValueError, match="fc"):
        fir_pasabajos(fc, 8000, 101)


@pytest.mark.parametrize("n_coef", [0, -3, 100, 101.0, True])
def test_pasabajos_rechaza_longitud_invalida(n_coef):
    with pytest.raises(ValueError, match="n_coef"):
        fir_pasabajos(1000, 8000, n_coef)


def test_pasabajos_rechaza_ventana_desconocida():
    with pytest.raises(ValueError, match="ventana"):
        fir_pasabajos(1000, 8000, 101, "desconocida")


@pytest.mark.parametrize("ventana", ["rect", "hann", "hamming"])
def test_pasaaltos_inversion_espectral(ventana):
    h = fir_pasaaltos(1000, 8000, 101, ventana)
    pasabajos = fir_pasabajos(1000, 8000, 101, ventana)
    impulso = np.zeros(101)
    impulso[50] = 1

    assert h.shape == (101,)
    assert np.isrealobj(h)
    assert np.all(np.isfinite(h))
    assert_allclose(h, h[::-1], atol=1e-15)
    assert_allclose(h + pasabajos, impulso, atol=1e-15)
    assert_allclose(np.sum(h), 0, atol=1e-14)


def test_pasaaltos_ganancia_y_atenuacion():
    h = fir_pasaaltos(1000, 8000, 101)
    # Se incluye Nyquist para comprobar también ese extremo de la banda de paso.
    freqs = np.linspace(0, 4000, 16385)
    _, respuesta = freqz(h, worN=freqs, fs=8000)
    amplitud = np.abs(respuesta)

    assert_allclose(amplitud[freqs >= 1300], 1, atol=0.01, rtol=0)
    assert np.max(amplitud[freqs <= 700]) < 0.01

    # firwin normaliza el pasaaltos en Nyquist; nuestra inversión conserva
    # la normalización del pasabajos en continua. Se comparan con tolerancia.
    referencia = firwin(101, 1000, fs=8000, pass_zero=False, window="hamming")
    _, esperado = freqz(referencia, worN=freqs, fs=8000)
    assert_allclose(respuesta, esperado, atol=0.005, rtol=0)


def test_pasaaltos_hamming_predeterminada():
    assert_allclose(
        fir_pasaaltos(1000, 8000, 101),
        fir_pasaaltos(1000, 8000, 101, "hamming"),
    )


def test_pasaaltos_un_coeficiente():
    # La inversión del pasabajos [1] produce [0], un caso degenerado.
    assert_allclose(fir_pasaaltos(1000, 8000, 1), [0.0])


@pytest.mark.parametrize(
    "fc, fs, n_coef, ventana, error",
    [
        (1000, 0, 101, "hamming", "fs"),
        (1000, np.inf, 101, "hamming", "fs"),
        (4000, 8000, 101, "hamming", "fc"),
        (np.nan, 8000, 101, "hamming", "fc"),
        (1000, 8000, 100, "hamming", "n_coef"),
        (1000, 8000, True, "hamming", "n_coef"),
        (1000, 8000, 101, "desconocida", "ventana"),
    ],
)
def test_pasaaltos_rechaza_entradas_invalidas(fc, fs, n_coef, ventana, error):
    with pytest.raises(ValueError, match=error):
        fir_pasaaltos(fc, fs, n_coef, ventana)


@pytest.mark.parametrize("ventana", ["rect", "hann", "hamming"])
def test_pasabanda_diferencia_de_pasabajos(ventana):
    h = fir_pasabanda(1000, 2000, 8000, 101, ventana)
    esperado = fir_pasabajos(2000, 8000, 101, ventana) - fir_pasabajos(1000, 8000, 101, ventana)

    assert h.shape == (101,)
    assert np.isrealobj(h)
    assert np.all(np.isfinite(h))
    assert_allclose(h, h[::-1], atol=1e-15)
    assert_allclose(np.sum(h), 0, atol=1e-14)
    assert_allclose(h, esperado, atol=1e-15)


def test_pasabanda_ganancia_y_atenuacion():
    h = fir_pasabanda(1000, 2000, 8000, 101)
    freqs = np.linspace(0, 4000, 16385)
    _, respuesta = freqz(h, worN=freqs, fs=8000)
    amplitud = np.abs(respuesta)

    # Se dejan 300 Hz de margen respecto a cada corte para excluir transiciones.
    paso = (freqs >= 1300) & (freqs <= 1700)
    assert_allclose(amplitud[paso], 1, atol=0.01, rtol=0)
    assert np.max(amplitud[freqs <= 700]) < 0.01
    assert np.max(amplitud[freqs >= 2300]) < 0.01

    # SciPy normaliza en el centro de la banda; nuestra diferencia conserva
    # la normalización de cada pasabajos. Se admite esa pequeña diferencia.
    referencia = firwin(101, [1000, 2000], fs=8000, pass_zero=False, window="hamming")
    _, esperado = freqz(referencia, worN=freqs, fs=8000)
    assert_allclose(respuesta, esperado, atol=0.005, rtol=0)


def test_pasabanda_hamming_predeterminada():
    assert_allclose(
        fir_pasabanda(1000, 2000, 8000, 101),
        fir_pasabanda(1000, 2000, 8000, 101, "hamming"),
    )


def test_pasabanda_un_coeficiente():
    assert_allclose(fir_pasabanda(1000, 2000, 8000, 1), [0.0])


@pytest.mark.parametrize(
    "f1, f2, fs, n_coef, ventana, error",
    [
        (1000, 2000, 0, 101, "hamming", "fs"),
        (1000, 2000, np.inf, 101, "hamming", "fs"),
        (0, 2000, 8000, 101, "hamming", "f1"),
        (-1000, 2000, 8000, 101, "hamming", "f1"),
        (np.nan, 2000, 8000, 101, "hamming", "f1"),
        (True, 2000, 8000, 101, "hamming", "f1"),
        (2000, 1000, 8000, 101, "hamming", "f2"),
        (1000, 1000, 8000, 101, "hamming", "f2"),
        (1000, 4000, 8000, 101, "hamming", "f2"),
        (1000, np.inf, 8000, 101, "hamming", "f2"),
        (1000, "2000", 8000, 101, "hamming", "f2"),
        (1000, 2000, 8000, 100, "hamming", "n_coef"),
        (1000, 2000, 8000, 0, "hamming", "n_coef"),
        (1000, 2000, 8000, 101, "desconocida", "ventana"),
    ],
)
def test_pasabanda_rechaza_entradas_invalidas(f1, f2, fs, n_coef, ventana, error):
    with pytest.raises(ValueError, match=error):
        fir_pasabanda(f1, f2, fs, n_coef, ventana)


@pytest.mark.parametrize("ventana", ["rect", "hann", "hamming"])
def test_rechazabanda_inversion_espectral(ventana):
    h = fir_rechazabanda(1000, 2000, 8000, 101, ventana)
    pasabanda = fir_pasabanda(1000, 2000, 8000, 101, ventana)
    impulso = np.zeros(101)
    impulso[50] = 1

    assert h.shape == (101,)
    assert np.isrealobj(h)
    assert np.all(np.isfinite(h))
    assert_allclose(h, h[::-1], atol=1e-15)
    assert_allclose(h + pasabanda, impulso, atol=1e-15)
    assert_allclose(np.sum(h), 1, atol=1e-14)


def test_rechazabanda_ganancia_y_atenuacion():
    h = fir_rechazabanda(1000, 2000, 8000, 101)
    freqs = np.linspace(0, 4000, 16385)
    _, respuesta = freqz(h, worN=freqs, fs=8000)
    amplitud = np.abs(respuesta)

    # Se excluyen las transiciones; las bandas de paso incluyen continua y Nyquist.
    assert_allclose(amplitud[freqs <= 700], 1, atol=0.01, rtol=0)
    assert_allclose(amplitud[freqs >= 2300], 1, atol=0.01, rtol=0)
    rechazo = (freqs >= 1300) & (freqs <= 1700)
    assert np.max(amplitud[rechazo]) < 0.01

    # La normalización de SciPy difiere ligeramente de invertir dos pasabajos
    # normalizados individualmente. Se compara toda la respuesta con tolerancia.
    referencia = firwin(101, [1000, 2000], fs=8000, pass_zero=True, window="hamming")
    _, esperado = freqz(referencia, worN=freqs, fs=8000)
    assert_allclose(respuesta, esperado, atol=0.005, rtol=0)


def test_rechazabanda_hamming_predeterminada():
    assert_allclose(
        fir_rechazabanda(1000, 2000, 8000, 101),
        fir_rechazabanda(1000, 2000, 8000, 101, "hamming"),
    )


def test_rechazabanda_un_coeficiente():
    # El pasabanda de un coeficiente es [0]; su inversión es la identidad [1].
    assert_allclose(fir_rechazabanda(1000, 2000, 8000, 1), [1.0])


@pytest.mark.parametrize(
    "f1, f2, fs, n_coef, ventana, error",
    [
        (1000, 2000, 0, 101, "hamming", "fs"),
        (1000, 2000, np.inf, 101, "hamming", "fs"),
        (0, 2000, 8000, 101, "hamming", "f1"),
        (np.nan, 2000, 8000, 101, "hamming", "f1"),
        (2000, 1000, 8000, 101, "hamming", "f2"),
        (1000, 1000, 8000, 101, "hamming", "f2"),
        (1000, 4000, 8000, 101, "hamming", "f2"),
        (1000, np.inf, 8000, 101, "hamming", "f2"),
        (1000, 2000, 8000, 100, "hamming", "n_coef"),
        (1000, 2000, 8000, True, "hamming", "n_coef"),
        (1000, 2000, 8000, 101, "desconocida", "ventana"),
    ],
)
def test_rechazabanda_rechaza_entradas_invalidas(f1, f2, fs, n_coef, ventana, error):
    with pytest.raises(ValueError, match=error):
        fir_rechazabanda(f1, f2, fs, n_coef, ventana)


def test_respuesta_frecuencia_impulso_y_malla_predeterminada():
    freqs, H = respuesta_frecuencia([1], 8000)

    assert freqs.shape == H.shape == (4096,)
    assert np.iscomplexobj(H)
    assert_allclose(freqs, np.linspace(0, 4000, 4096))
    assert_allclose(H, 1, atol=1e-15)


def test_respuesta_frecuencia_impulso_retardado():
    freqs, H = respuesta_frecuencia([0, 0, 1], 8000, n=17)

    # Un retardo de dos muestras tiene módulo uno y fase -2 * omega.
    esperado = np.exp(-1j * 2 * np.pi * freqs * 2 / 8000)
    assert_allclose(H, esperado, atol=1e-14)
    assert_allclose(np.abs(H), 1, atol=1e-14)


@pytest.mark.parametrize("longitud", [1, 2, 17, 101])
@pytest.mark.parametrize("complejo", [False, True])
def test_respuesta_frecuencia_coincide_con_scipy(longitud, complejo):
    rng = np.random.default_rng(2026)
    h = rng.normal(size=longitud)
    if complejo:
        h = h + 1j * rng.normal(size=longitud)
    original = h.copy()

    freqs, H = respuesta_frecuencia(h, 8000, n=257)
    _, esperado = freqz(h, worN=freqs, fs=8000)

    assert_allclose(H, esperado, atol=1e-11, rtol=1e-12)
    assert_allclose(h, original, atol=0, rtol=0)


def test_respuesta_frecuencia_promedio_de_dos_muestras():
    freqs, H = respuesta_frecuencia([0.5, 0.5], 8000, n=3)

    assert_allclose(freqs, [0, 2000, 4000])
    assert_allclose(H, [1, 0.5 - 0.5j, 0], atol=1e-15)


def test_respuesta_frecuencia_un_punto():
    freqs, H = respuesta_frecuencia([1, 2j, 3], 8000, n=1)

    assert_allclose(freqs, [0])
    assert_allclose(H, [4 + 2j])


@pytest.mark.parametrize("h", [[], [[1, 2]], 1, [np.nan], [np.inf], [1 + np.inf * 1j], ["a"], [True]])
def test_respuesta_frecuencia_rechaza_coeficientes_invalidos(h):
    with pytest.raises(ValueError, match="h"):
        respuesta_frecuencia(h, 8000)


@pytest.mark.parametrize("fs", [0, -8000, np.nan, np.inf, True, "8000"])
def test_respuesta_frecuencia_rechaza_fs_invalida(fs):
    with pytest.raises(ValueError, match="fs"):
        respuesta_frecuencia([1], fs)


@pytest.mark.parametrize("n", [0, -1, 17.0, True])
def test_respuesta_frecuencia_rechaza_n_invalido(n):
    with pytest.raises(ValueError, match="n"):
        respuesta_frecuencia([1], 8000, n)


@pytest.mark.parametrize("longitud", [1, 3, 30])
@pytest.mark.parametrize("complejo", [False, True])
def test_filtrar_coincide_con_scipy(longitud, complejo):
    rng = np.random.default_rng(2026)
    x = rng.normal(size=longitud)
    if complejo:
        x = x + 1j * rng.normal(size=longitud)
    h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    original_x, original_h = x.copy(), h.copy()

    y = filtrar(x, h)
    esperado = convolve(x, h, mode="same", method="direct")

    assert y.shape == x.shape
    assert_allclose(y, esperado, atol=1e-14)
    assert_allclose(x, original_x, atol=0, rtol=0)
    assert_allclose(h, original_h, atol=0, rtol=0)


@pytest.mark.parametrize(
    "h",
    [
        fir_pasabajos(1000, 8000, 101),
        fir_pasaaltos(1000, 8000, 101),
        fir_pasabanda(1000, 2000, 8000, 101),
        fir_rechazabanda(1000, 2000, 8000, 101),
    ],
    ids=["pasabajos", "pasaaltos", "pasabanda", "rechazabanda"],
)
def test_filtrar_impulso_sin_desplazamiento(h):
    x = np.zeros(401)
    x[200] = 1

    y = filtrar(x, h)
    esperado = np.zeros_like(x)
    esperado[150:251] = h

    assert_allclose(y, esperado, atol=1e-15)
    assert np.argmax(np.abs(y)) == 200


def test_filtrar_bordes_con_ceros():
    # El promedio conserva la constante en el interior y cae en los extremos.
    assert_allclose(filtrar([1, 1, 1, 1, 1], [0.25, 0.5, 0.25]), [0.75, 1, 1, 1, 0.75])


def test_filtrar_identidad_y_enteros():
    x = np.array([10000, -10000, 20000], dtype=np.int16)
    assert_allclose(filtrar(x, [1]), x)
    assert_allclose(filtrar(x, np.array([10000], dtype=np.int16)), [1e8, -1e8, 2e8])


@pytest.mark.parametrize(
    "h, frecuencia_paso, frecuencia_rechazo",
    [
        (fir_pasabajos(1000, 8000, 101), 500, 2500),
        (fir_pasaaltos(1000, 8000, 101), 2500, 500),
        (fir_pasabanda(1000, 2000, 8000, 101), 1500, 500),
        (fir_rechazabanda(1000, 2000, 8000, 101), 500, 1500),
    ],
    ids=["pasabajos", "pasaaltos", "pasabanda", "rechazabanda"],
)
def test_filtrar_conserva_tono_de_paso_y_atenua_otro(h, frecuencia_paso, frecuencia_rechazo):
    t = np.arange(4096) / 8000
    deseada = np.sin(2 * np.pi * frecuencia_paso * t)
    x = deseada + np.sin(2 * np.pi * frecuencia_rechazo * t)

    y = filtrar(x, h)

    # Fuera de los bordes, debe quedar el tono deseado sin desfase temporal.
    assert_allclose(y[50:-50], deseada[50:-50], atol=0.015, rtol=0)


@pytest.mark.parametrize("x", [[], [[1, 2]], 1, [np.nan], [np.inf], ["a"], [True]])
def test_filtrar_rechaza_senal_invalida(x):
    with pytest.raises(ValueError, match="x"):
        filtrar(x, [1])


@pytest.mark.parametrize("h", [[], [[1]], 1, [np.nan], [np.inf], [1j], ["a"], [True], [0.5, 0.5], [1, 2, 3]])
def test_filtrar_rechaza_filtro_invalido(h):
    with pytest.raises(ValueError, match="h"):
        filtrar([1, 2, 3], h)


@pytest.mark.parametrize("n_coef", [21, 51, 101, 201])
@pytest.mark.parametrize(
    "ventana, referencia",
    [("rect", "boxcar"), ("hamming", "hamming"), ("hann", "hann")],
)
def test_pasabajos_disenos_del_enunciado(n_coef, ventana, referencia):
    h = fir_pasabajos(100, 1000, n_coef, ventana)
    esperado = firwin(n_coef, 100, fs=1000, window=referencia)

    assert h.shape == (n_coef,)
    assert_allclose(h, h[::-1], atol=1e-15)
    assert_allclose(np.sum(h), 1, atol=1e-14)
    assert_allclose(h, esperado, atol=1e-15, rtol=1e-12)

    # Se contrasta también nuestra respuesta, no solo los coeficientes.
    freqs, H = respuesta_frecuencia(h, 1000)
    _, H_referencia = freqz(esperado, worN=freqs, fs=1000)
    assert_allclose(H, H_referencia, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n_coef", [101, 201])
def test_pasabajos_hamming_rechazo_del_enunciado(n_coef):
    h = fir_pasabajos(100, 1000, n_coef)
    freqs, H = respuesta_frecuencia(h, 1000)
    amplitud = np.abs(H)

    # Para estas longitudes, 50 Hz de margen excluyen las transiciones.
    # No se exige este rechazo para cualquier longitud ni para otras ventanas.
    assert_allclose(amplitud[freqs <= 50], 1, atol=0.01, rtol=0)
    assert np.max(amplitud[freqs >= 150]) < 0.01


def test_experimento_filtros_genera_resultados_validos(tmp_path):
    import matplotlib.pyplot as plt

    from psi_lab.lab1.experimentos.exp_filtros import generar

    figuras_previas = plt.get_fignums()
    destino = tmp_path / "resultados_fir"
    generar(str(destino))

    assert plt.get_fignums() == figuras_previas
    for tarea, nombres in [
        ("tarea4_2", ["respuestas_frecuencia.png", "respuestas_frecuencia.pdf"]),
        ("tarea4_3", ["chirp_filtrado.png", "chirp_filtrado.pdf", "detalle_corte.png"]),
    ]:
        assert (destino / tarea / "resultados.md").is_file()
        for nombre in nombres:
            datos = (destino / tarea / nombre).read_bytes()
            firma = b"\x89PNG\r\n\x1a\n" if nombre.endswith(".png") else b"%PDF-"
            assert datos.startswith(firma)

    with (destino / "tarea4_2" / "mediciones.csv").open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert len(filas) == 12
    assert {(fila["ventana"], int(fila["n_coef"])) for fila in filas} == {
        (ventana, n_coef) for ventana in ["Rectangular", "Hamming", "Hann"] for n_coef in [21, 51, 101, 201]
    }
    for fila in filas:
        ventana = {"Rectangular": "boxcar", "Hamming": "hamming", "Hann": "hann"}[fila["ventana"]]
        h = firwin(int(fila["n_coef"]), 100, fs=1000, window=ventana)
        cruces = np.array([float(fila["f_90_Hz"]), float(fila["f_10_Hz"])])
        _, H = freqz(h, worN=cruces, fs=1000)
        assert_allclose(np.abs(H), [0.9, 0.1], atol=2e-6, rtol=0)
        assert_allclose(float(fila["transicion_Hz"]), cruces[1] - cruces[0])
        assert 100 < float(fila["inicio_rechazo_Hz"]) < 500
        assert float(fila["inicio_rechazo_Hz"]) > cruces[1]
        freqs = np.linspace(0, 500, 32769)
        _, H = freqz(h, worN=freqs, fs=1000)
        maximo_db = 20 * np.log10(np.max(np.abs(H[freqs >= float(fila["inicio_rechazo_Hz"])])))
        assert_allclose(float(fila["max_rechazo_dB"]), maximo_db, atol=1e-9)
        assert_allclose(float(fila["atenuacion_dB"]), -maximo_db, atol=1e-9)

    with (destino / "tarea4_3" / "mediciones.csv").open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert [fila["region"] for fila in filas] == ["Paso", "Alrededor del corte", "Rechazo"]
    for fila in filas:
        rms_entrada, rms_salida = float(fila["rms_entrada"]), float(fila["rms_salida"])
        assert np.isfinite(rms_entrada) and np.isfinite(rms_salida)
        assert rms_entrada > 0 and rms_salida > 0
        assert_allclose(float(fila["cambio_rms_dB"]), 20 * np.log10(rms_salida / rms_entrada))
    assert abs(float(filas[0]["cambio_rms_dB"])) < 0.1
    assert -10 < float(filas[1]["cambio_rms_dB"]) < -2
    assert float(filas[2]["cambio_rms_dB"]) < -40
