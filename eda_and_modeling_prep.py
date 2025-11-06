import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuración ---
cleaned_data_path = "siniestros_viales_limpios.csv"
eda_report_path = "eda_report.md"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# --- 1. Carga de Datos Limpios ---
print(f"Cargando datos limpios desde: {cleaned_data_path}...")
df = pd.read_csv(cleaned_data_path)

# Convertir columnas de fecha/hora que se guardaron como strings
df['fecha_hora_accidente'] = pd.to_datetime(df['fecha_hora_accidente'])
df['fecha_accidente'] = pd.to_datetime(df['fecha_accidente'])
df['hora_accidente'] = pd.to_datetime(df['hora_accidente']).dt.time

# --- 2. Análisis Exploratorio de Datos (EDA) ---

eda_content = "## Análisis Exploratorio de Datos (EDA)\n\n"
eda_content += "### 2.1. Distribución de la Variable Objetivo (Gravedad)\n"
eda_content += "La variable objetivo para el modelo predictivo es `gravedad_binaria` (1: Grave - Muerto/Lesionado, 0: Leve - Solo Daños).\n"

# 2.1. Distribución de la Variable Objetivo
gravedad_counts = df['gravedad_binaria'].value_counts(normalize=True) * 100
eda_content += "La distribución de la gravedad muestra un problema de desbalance de clases:\n"
eda_content += f"| Gravedad | Porcentaje |\n| :--- | :--- |\n| Grave (1) | {gravedad_counts.loc[1]:.2f}% |\n| Leve (0) | {gravedad_counts.loc[0]:.2f}% |\n"

# 2.2. Análisis Temporal
eda_content += "\n### 2.2. Análisis Temporal\n"

# Siniestros por Día de la Semana
plt.figure(figsize=(10, 6))
order_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
sns.countplot(y='dia_semana', data=df, order=order_dias, palette="viridis")
plt.title('Siniestros por Día de la Semana')
plt.xlabel('Número de Siniestros')
plt.ylabel('Día de la Semana')
plt.savefig(os.path.join(plots_dir, 'siniestros_por_dia_semana.png'))
plt.close()
eda_content += "Se generó el gráfico 'siniestros_por_dia_semana.png' mostrando la distribución de siniestros por día de la semana.\n"

# Siniestros por Hora del Día
plt.figure(figsize=(12, 6))
sns.countplot(x='hora_del_dia', data=df, palette="magma")
plt.title('Siniestros por Hora del Día')
plt.xlabel('Hora del Día (0-23)')
plt.ylabel('Número de Siniestros')
plt.savefig(os.path.join(plots_dir, 'siniestros_por_hora_dia.png'))
plt.close()
eda_content += "Se generó el gráfico 'siniestros_por_hora_dia.png' mostrando la distribución de siniestros por hora del día.\n"

# 2.3. Análisis Geográfico (Localidad)
eda_content += "\n### 2.3. Análisis Geográfico (Localidad)\n"

# Siniestros por Localidad (usando codigo_localidad)
plt.figure(figsize=(12, 8))
localidad_counts = df['codigo_localidad'].value_counts().sort_values(ascending=False)
sns.barplot(x=localidad_counts.index, y=localidad_counts.values, palette="rocket")
plt.title('Siniestros por Código de Localidad')
plt.xlabel('Código de Localidad')
plt.ylabel('Número de Siniestros')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'siniestros_por_localidad.png'))
plt.close()
eda_content += "Se generó el gráfico 'siniestros_por_localidad.png' mostrando la distribución de siniestros por código de localidad.\n"

# 2.4. Relación entre Variables y Gravedad
eda_content += "\n### 2.4. Relación entre Variables y Gravedad\n"

# Gravedad por Día de la Semana
plt.figure(figsize=(10, 6))
sns.countplot(y='dia_semana', hue='gravedad_binaria', data=df, order=order_dias, palette="coolwarm")
plt.title('Gravedad de Siniestros por Día de la Semana')
plt.xlabel('Número de Siniestros')
plt.ylabel('Día de la Semana')
plt.legend(title='Gravedad', labels=['Leve (0)', 'Grave (1)'])
plt.savefig(os.path.join(plots_dir, 'gravedad_por_dia_semana.png'))
plt.close()
eda_content += "Se generó el gráfico 'gravedad_por_dia_semana.png' para comparar la gravedad por día de la semana.\n"

# --- 3. Preparación Final para el Modelado ---

# 3.1. Selección de Características (Features)
# Variables predictoras (X):
# - dia_semana, hora_del_dia, codigo_localidad, clase, diseno_lugar
# Variable objetivo (y):
# - gravedad_binaria

features = ['dia_semana', 'hora_del_dia', 'codigo_localidad', 'clase', 'diseno_lugar']
target = 'gravedad_binaria'

# 3.2. Manejo de Variables Categóricas (One-Hot Encoding)
df_model = df[features + [target]].copy()

# Convertir variables categóricas a tipo 'category' para un manejo eficiente
categorical_features = ['dia_semana', 'codigo_localidad', 'clase', 'diseno_lugar']
for col in categorical_features:
    df_model[col] = df_model[col].astype('category')

# Aplicar One-Hot Encoding
df_model = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

# 3.3. Guardar el DataFrame listo para el modelado
modeling_data_path = "siniestros_viales_modelado.csv"
df_model.to_csv(modeling_data_path, index=False)
eda_content += f"\n### 3. Preparación para el Modelado\n"
eda_content += f"El DataFrame final para el modelado, con One-Hot Encoding aplicado, se guardó en: {modeling_data_path}\n"
eda_content += f"El DataFrame tiene {df_model.shape[0]} filas y {df_model.shape[1]} columnas (incluyendo la variable objetivo)."

# --- 4. Guardar Reporte EDA ---
with open(eda_report_path, "w") as f:
    f.write(eda_content)

print(f"\nReporte EDA guardado en: {eda_report_path}")
print(f"Datos listos para el modelado guardados en: {modeling_data_path}")
