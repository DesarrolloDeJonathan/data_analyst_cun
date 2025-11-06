import pandas as pd
import os

# --- Configuración ---
# El archivo se descargó en el directorio Downloads
file_path = os.path.expanduser("~/Downloads/siniestros_viales_consolidados_bogota_dc.xlsx")
cleaned_data_path = "siniestros_viales_limpios.csv"
metadata_path = "data_metadata.txt"

# --- 1. Recolección de Datos (Lectura) ---
print(f"Leyendo el archivo: {file_path}...")
try:
    # Leer la primera hoja del archivo XLSX
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Lectura exitosa.")
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit()

# --- 2. Inspección Inicial y Documentación ---
print("\nRealizando inspección inicial...")

# Guardar la información del DataFrame en un archivo de metadatos
with open(metadata_path, "w") as f:
    f.write("--- Información General del DataFrame ---\n")
    df.info(buf=f)
    f.write("\n\n--- Primeras 5 Filas ---\n")
    f.write(df.head().to_string())
    f.write("\n\n--- Conteo de Valores Nulos ---\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\n--- Conteo de Valores Únicos ---\n")
    f.write(df.nunique().to_string())

print(f"Metadatos iniciales guardados en: {metadata_path}")

# --- 3. Limpieza y Transformación (Preparación para el EDA) ---

# Convertir nombres de columnas a minúsculas y reemplazar caracteres especiales
df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False).str.replace('á', 'a').str.replace('é', 'e').str.replace('í', 'i').str.replace('ó', 'o').str.replace('ú', 'u').str.replace('.', '', regex=False).str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.replace('-', '_', regex=False).str.replace('/', '_', regex=False)

# Convertir la columna de fecha/hora
# Las columnas son 'fecha' y 'hora' (en minúsculas por el pre-procesamiento anterior)
if 'fecha' in df.columns and 'hora' in df.columns:
    # Combinar las columnas 'fecha' y 'hora' y convertirlas a datetime
    df['fecha_hora_accidente'] = pd.to_datetime(df['fecha'] + ' ' + df['hora'].astype(str), errors='coerce')
    df['fecha_accidente'] = df['fecha_hora_accidente'].dt.date
    df['hora_accidente'] = df['fecha_hora_accidente'].dt.time
    # Creación de variables temporales clave para el modelo predictivo
    # Calcular el día de la semana (0=Lunes, 6=Domingo) y mapear a español
    dias_semana = {
        0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
        4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
    }
    df['dia_semana'] = df['fecha_hora_accidente'].dt.dayofweek.map(dias_semana)
    df['mes'] = df['fecha_hora_accidente'].dt.month
    df['anio'] = df['fecha_hora_accidente'].dt.year
    df['hora_del_dia'] = df['fecha_hora_accidente'].dt.hour
    
    # --- Definición de la Variable Objetivo ---
    # La columna 'gravedad' tiene 3 valores únicos (45: GRAVEDAD 3). Asumiremos:
    # 1: Solo Daños (Leve)
    # 2: Lesionado (Grave)
    # 3: Muerto (Grave)
    # Variable objetivo: 1 = Grave (Muerto o Lesionado), 0 = Leve (Solo Daños)
    df['gravedad_binaria'] = df['gravedad'].apply(lambda x: 1 if x in [2, 3] else 0)

    print(f"Variable objetivo 'gravedad_binaria' creada. Distribución: \n{df['gravedad_binaria'].value_counts()}")
    
# Guardar el DataFrame pre-procesado a CSV
print("\nGuardando el DataFrame pre-procesado a CSV...")
df.to_csv(cleaned_data_path, index=False)
print(f"DataFrame guardado en: {cleaned_data_path}")

# Imprimir un resumen de las columnas para el siguiente paso
print("\nColumnas del DataFrame después de la limpieza:")
print(df.columns.tolist())
