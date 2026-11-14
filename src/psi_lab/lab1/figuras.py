import os

from psi_lab.lab1.experimentos import (
exp_caso,
exp_convolucion,
exp_dft,
exp_filtros,
exp_muestreo,
)

"""Para ejecutar todas las tablas y fuguras"""

def main(out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)
    for exp in [exp_muestreo, exp_dft, exp_convolucion, exp_filtros, exp_caso]:
        exp.generar(out_dir)

if __name__ == "__main__":
    main()