import numpy as np

"""
Funcion muestrear : Calcula los instantes tn y los evalua con fun
Entrada: fun (funcion matematica que se va a evaluar)
         fs (frecuencia de muestreo)
         duracion (duracion de señal en segundos)
"""
def muestrear(fun, fs, duracion):

    # Comprobar argumentos validos para muestreo
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser positiva y finita")

    if not np.isfinite(duracion) or duracion < 0:
        raise ValueError("duration debe ser mayor a 0 y finita")

    # Generar indices
    n = np.arange(int(np.ceil(fs * duracion)))

    # Convertir indices en tiempos que esten dentro del intervalo
    t = n / fs
    t = t[t < duracion]

    # Evaluar funcion en cada instante
    x = np.asarray([fun(ti) for ti in t])

    return t, x


"""
Funcion frecuencia_alias(f, fs) : calcula la frecuencia aparente de una señal muestreada
Entrada: f (frecuencia original)
         fs (frecuencia de muestreo)
"""

def frecuencia_alias(f, fs):

    # Comprobar argumentos validos
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser positiva y finita")

    if not np.isfinite(f):
        raise ValueError("f debe ser finito")

    # Ejecutar formula
    alias = abs((f + fs/2) % fs - fs / 2)

    return alias