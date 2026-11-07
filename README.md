# PSI 2-2026 - Laboratorio 1

Grupo de trabajo para el curso Procesamiento de Señales e Imágenes (13318), USACH.

## Estructura del Proyecto
- `src/psi_lab/lab1/`: Herramientas de la Parte Común y Caso C (Imágenes).
- `tests/lab1/`: Pruebas unitarias y de integración.
- `informe/`: Informe técnico breve (`lab1.pdf`).

## Instalación Local
```bash
python3 -m venv .venv
source .venv/bin/activate  

# En Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"