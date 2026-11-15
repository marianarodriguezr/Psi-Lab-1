
import numpy as np
import pytest

from psi_lab.lab1.muestreo import frecuencia_alias, muestrear


def test_muestrear():
    """Comprueba los instantes y valores del muestreo."""

    fs = 100
    duracion = 1
    f = 10

    fun = lambda t: np.sin(2 * np.pi * f * t)
    t , x = muestrear(fun, fs, duracion)

    # Deben existir 100 muestras.
    assert len(t) == 100
    assert len(x) == 100

    # Los tiempos deben ser correctos.
    t_esperado = np.arange(100) / fs
    np.testing.assert_allclose(t, t_esperado)

    # Los valores deben coincidir con la señal original.
    x_esperado = np.sin(2 * np.pi * f * t)
    np.testing.assert_allclose(x, x_esperado)


def test_frecuencia_alias():
    """Comprueba frecuencias aparentes conocidas."""

    assert frecuencia_alias(10, 100) == 10
    assert frecuencia_alias(90, 100) == 10
    assert frecuencia_alias(110, 100) == 10
    assert frecuencia_alias(200, 100) == 0
    assert frecuencia_alias(250, 100) == 50


def test_parametros_invalidos():
    """Comprueba que se rechacen frecuencias de muestreo invalidas."""

    with pytest.raises(ValueError):
        muestrear(lambda t: t, 0, 1)

    with pytest.raises(ValueError):
        frecuencia_alias(10, -100)


def test_aliasing_frecuencias_aleatorias():
    """Comprueba el pico espectral usando frecuencias aleatorias."""

    rng = np.random.default_rng(2026)

    fs = 256
    duracion = 1

    # Se generan 50 frecuencias aleatorias enteras.
    frecuencias = rng.integers(1, 4 * fs, size=50)

    for f in frecuencias:

        # Calculamos la frecuencia alias teorica.
        f_alias = frecuencia_alias(f, fs)

        # Excluimos los casos especiales de DC y Nyquist.
        if f_alias == 0 or f_alias == fs / 2:
            continue

        # Generamos y muestreamos la señal.
        fun = lambda t, f=f: np.sin(2 * np.pi * f * t + 0.23)
        _, x = muestrear(fun, fs, duracion)

        # Calculamos su espectro mediante FFT.
        espectro = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1 / fs)

        # Buscamos el pico espectral dominante.
        indice_pico = np.argmax(espectro[1:]) + 1
        frecuencia_detectada = freqs[indice_pico]

        # Debe coincidir con la frecuencia alias teorica.
        assert frecuencia_detectada == f_alias
