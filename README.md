# Proyecto: Análisis de siniestros viales (data_analitys_cun)

Este repositorio contiene un flujo de trabajo completo para la limpieza, análisis exploratorio, preparación de datos, modelado y despliegue ligero (dashboard) sobre el dataset `siniestros_viales_limpios.csv`.

El README siguiente está escrito paso a paso para desarrolladores que trabajan en Windows usando `cmd.exe` y asume una instalación de Python >= 3.10.

## Contenido del repositorio

- `app_dashboard.py` — script para lanzar un dashboard (visualización interactiva).
- `check_env.py` — comprobaciones rápidas del entorno (paquetes, versiones, disponibilidades).
- `data_preparation.py` — script para limpieza y transformación de los datos crudos.
- `eda_and_modeling_prep.py` — análisis exploratorio y preparación final para modelado.
- `modeling.py` — entrenamientos y evaluación de modelos.
- `main.py` — flujo principal (orquestador) que ejecuta los pasos por defecto del pipeline.
- `pyproject.toml` — metadatos del proyecto (opcional).
- `requirements.txt` — dependencias pinneadas para instalar con pip.
- `siniestros_viales_limpios.csv` — dataset de entrada (limpio / pre-filtrado).

## Contrato mínimo (entradas / salidas / errores)

- Entrada principal: `siniestros_viales_limpios.csv` (CSV con columnas del registro de siniestros).
- Salidas esperadas: archivos procesados en `data/` (por ejemplo `data/processed.csv`), modelos serializados en `models/`, reportes/figuras en `reports/` y un dashboard ejecutable.
- Modo de fallo típico: dependencias no instaladas, paths relativos incorrectos, o CSV con columnas faltantes. Los scripts deberían generar mensajes claros en `stderr`.

Edge cases a considerar:

- Archivo de datos vacío o con encoding distinto a UTF-8.
- Columnas faltantes o tipos inconsistentes (fechas mal formateadas, strings en columnas numéricas).
- Falta de memoria si se carga un dataset muy grande en memoria.

## Requisitos

- Python 3.10+ (preferible 3.11)
- Pip
- Herramientas: git (opcional)

Las dependencias se listan en `requirements.txt`. Si quieres usar `pyproject.toml`/poetry, adáptalo según prefieras.

## Instalación (Windows / cmd.exe)

1. Crear y activar un entorno virtual (recomendado):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Instalar dependencias:

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

3. (Opcional) Si usas `pyproject.toml` con Poetry, sigue tu flujo de Poetry en lugar del paso anterior.

## Verificar el entorno

Ejecuta `check_env.py` para comprobar que las librerías principales estén instaladas y las versiones sean compatibles. Este script imprime advertencias y recomendaciones.

```cmd
python check_env.py
```

Nota: Si `check_env.py` arroja errores de dependencias, instala los paquetes faltantes con `pip`.

## Preparación de datos

El flujo típico de preparación es:

1. Ejecutar `data_preparation.py` para limpiar y estandarizar el CSV.
2. Ejecutar `eda_and_modeling_prep.py` para obtener análisis exploratorio y preparar las variables finales para el modelo.

Ejemplos (asumimos que los scripts aceptan argumentos; si no, ejecuta sin argumentos — revisa los docstrings en cada archivo):

```cmd
python data_preparation.py --input siniestros_viales_limpios.csv --output data/processed.csv
python eda_and_modeling_prep.py --input data/processed.csv --output data/model_input.csv
```

Si los scripts no exponen CLI, ejecútalos directamente y revisa/edita las variables `INPUT_PATH` / `OUTPUT_PATH` dentro de cada archivo.

## Modelado

Usa `modeling.py` para entrenar y evaluar modelos. El script debe generar artefactos en `models/` y métricas en `reports/`.

Ejemplo:

```cmd
python modeling.py --train data/model_input.csv --out models/ --reports reports/
```

Si `modeling.py` no soporta argumentos, abre el archivo para ver las rutas y parámetros configurables.

## Ejecutar todo el pipeline

El `main.py` está pensado como orquestador mínimo que corre los pasos en orden: preparación, EDA, modelado y (opcional) dashboard.

```cmd
python main.py
```

Revisa `main.py` para ver flags disponibles (modo debug, rutas, etc.).

## Dashboard

Para ver visualizaciones interactivas ejecuta:

```cmd
python app_dashboard.py
```

Dependiendo de la implementación, esto puede lanzar un servidor local (por ejemplo `http://127.0.0.1:8050`).

## Estructura recomendada de carpetas (si no existen crea):

- `data/` — datos crudos y procesados.
- `models/` — modelos entrenados (.pkl, .joblib o el formato elegido).
- `reports/` — gráficos, métricas y reportes.
- `notebooks/` — análisis exploratorio adicional en notebooks (opcional).

## Buenas prácticas y notas

- Haz commits pequeños y con mensajes claros cuando cambies scripts de preparación o modelado.
- Versiona tus modelos (ej. `models/model_v1.pkl`).
- Documenta los cambios en las transformaciones (por ejemplo, guardar `data/transformer_config.json`).

## Cómo contribuir

1. Haz fork / branch.
2. Añade tests o verificaciones mínimas para cambios que modifiquen lógica del pipeline.
3. Envía pull request describiendo el cambio y su impacto.

## Resolución de problemas comunes

- Error de importación: asegúrate de haber activado `.venv` e instalado `requirements.txt`.
- Error de codificación al leer CSV: abre el CSV con `encoding='utf-8'` o detecta encoding con `chardet`.
- Rutas relativas: si usas `cmd.exe`, recuerda que `.` es la carpeta actual; usa rutas absolutas si tienes dudas.

## Próximos pasos sugeridos

- Añadir tests unitarios mínimos (por ejemplo en `tests/test_data_preparation.py`).
- Añadir un pequeño script de CI (GitHub Actions) que ejecute `check_env.py` y una corrida rápida de `main.py` con un subset de datos.

## Licencia y contacto

Incluye aquí la licencia del proyecto (por ejemplo MIT) y datos de contacto si corresponde.

---

Si quieres, puedo también:

- Añadir un archivo `CONTRIBUTING.md` con plantilla de PR y guía para estilo de código.
- Crear tests mínimos para `data_preparation.py`.

Indica si deseas que actualice `requirements.txt` automáticamente o que añada un ejemplo de `data/` con un subset para pruebas rápidas.
