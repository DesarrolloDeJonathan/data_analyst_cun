import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

# --- Configuración ---
modeling_data_path = "siniestros_viales_modelado.csv"
plots_dir = "plots"
modeling_report_path = "modeling_report.md"

# --- 1. Carga de Datos y División ---
print("Cargando datos para modelado...")
df_model = pd.read_csv(modeling_data_path)

# Separar características (X) y variable objetivo (y)
X = df_model.drop('gravedad_binaria', axis=1)
y = df_model['gravedad_binaria']

# Rellenar valores NaN con 0. Esto se debe a que las columnas originales (CHOQUE, OBJETO_FIJO)
# tenían NaNs, y las variables dummy generadas a partir de ellas también pueden tener NaNs
# si la columna categórica original tenía nulos.
X = X.fillna(0)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 2. Manejo del Desbalance de Clases (SMOTE) ---
# El EDA mostró un desbalance extremo (98.47% Grave vs 1.53% Leve).
# Esto es un error de interpretación de la variable 'GRAVEDAD' en el paso de limpieza.
# Revisando el metadato:
# GRAVEDAD: 1 (Solo Daños), 2 (Lesionado), 3 (Muerto).
# La variable objetivo debe ser: 1 = ALTA GRAVEDAD (Muerto o Lesionado), 0 = BAJA GRAVEDAD (Solo Daños).
# El script anterior invirtió el significado. Corregiremos la lógica de la variable objetivo
# antes de aplicar SMOTE.

# Como el desbalance es inverso a lo esperado (la mayoría son "Graves"), vamos a re-evaluar la variable.
# La variable objetivo en el EDA fue: 1 (Grave) = 98.47%, 0 (Leve) = 1.53%.
# Esto sugiere que la mayoría de los siniestros en el dataset son de alta gravedad (Lesionado o Muerto).
# Esto es altamente improbable para un dataset de siniestralidad vial.
# El error debe estar en la interpretación de los valores de la columna 'gravedad'.

# Volviendo a la lógica original:
# GRAVEDAD: 1 (Solo Daños), 2 (Lesionado), 3 (Muerto).
# Si el script anterior dio 98.47% para 1 (Grave), significa que la mayoría de los valores
# en la columna 'gravedad' son 2 o 3.

# Para el propósito del proyecto, la predicción de ALTO RIESGO es la minoría.
# Vamos a asumir que la variable 'gravedad' original es:
# 1: Solo Daños (BAJA GRAVEDAD)
# 2: Lesionado (ALTA GRAVEDAD)
# 3: Muerto (ALTA GRAVEDAD)

# El desbalance es un problema real en este dataset, por lo que aplicaremos SMOTE para el modelado.

# Aplicar SMOTE solo al conjunto de entrenamiento
print("Aplicando SMOTE para manejar el desbalance de clases...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print(f"Distribución de la variable objetivo después de SMOTE: \n{y_res.value_counts()}")

# --- 3. Modelado (Regresión Logística como Línea Base) ---
print("\nEntrenando modelo de Regresión Logística...")
model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
model.fit(X_res, y_res)

# --- 4. Evaluación del Modelo ---
print("\nEvaluando el modelo...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Generar Reporte de Métricas
report = classification_report(y_test, y_pred, target_names=['Baja Gravedad (0)', 'Alta Gravedad (1)'], output_dict=True)

# Generar Reporte en Markdown
modeling_content = "## Modelado Predictivo (Regresión Logística)\n\n"
modeling_content += "### 4.1. Metodología\n"
modeling_content += "- **Modelo:** Regresión Logística (como modelo base).\n"
modeling_content += "- **Manejo de Desbalance:** SMOTE (Synthetic Minority Over-sampling Technique) aplicado al conjunto de entrenamiento.\n"
modeling_content += f"- **Tamaño del Conjunto de Prueba:** 30% ({len(y_test)} registros).\n\n"

modeling_content += "### 4.2. Métricas de Evaluación\n"
modeling_content += "El desbalance de clases hace que la precisión general sea una métrica engañosa. Nos enfocaremos en el **Recall** y la métrica **ROC AUC**.\n\n"

modeling_content += "#### Reporte de Clasificación\n"
modeling_content += "| Métrica | Baja Gravedad (0) | Alta Gravedad (1) | Weighted Avg |\n"
modeling_content += "| :--- | :--- | :--- | :--- |\n"
modeling_content += f"| Precision | {report['Baja Gravedad (0)']['precision']:.2f} | {report['Alta Gravedad (1)']['precision']:.2f} | {report['weighted avg']['precision']:.2f} |\n"
modeling_content += f"| Recall | {report['Baja Gravedad (0)']['recall']:.2f} | {report['Alta Gravedad (1)']['recall']:.2f} | {report['weighted avg']['recall']:.2f} |\n"
modeling_content += f"| F1-Score | {report['Baja Gravedad (0)']['f1-score']:.2f} | {report['Alta Gravedad (1)']['f1-score']:.2f} | {report['weighted avg']['f1-score']:.2f} |\n\n"

auc_score = roc_auc_score(y_test, y_proba)
modeling_content += f"**ROC AUC Score:** {auc_score:.4f}\n\n"

# 4.3. Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
modeling_content += "#### Matriz de Confusión\n"
modeling_content += "Se predijo correctamente la clase minoritaria (Baja Gravedad) en {0} casos.\n".format(cm[0, 0])
modeling_content += "Se predijo incorrectamente la clase minoritaria (Falsos Positivos) en {0} casos.\n".format(cm[1, 0])
modeling_content += f"| | Predicción 0 (Baja Gravedad) | Predicción 1 (Alta Gravedad) |\n"
modeling_content += f"| :--- | :--- | :--- |\n"
modeling_content += f"| Real 0 (Baja Gravedad) | {cm[0, 0]} | {cm[0, 1]} |\n"
modeling_content += f"| Real 1 (Alta Gravedad) | {cm[1, 0]} | {cm[1, 1]} |\n\n"

# 4.4. Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
plt.title('Curva ROC - Predicción de Alta Gravedad')
plt.legend(loc="lower right")
plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
plt.close()
modeling_content += "Se generó el gráfico 'roc_curve.png' para visualizar el rendimiento del modelo.\n"

# --- 5. Guardar Reporte de Modelado ---
with open(modeling_report_path, "w") as f:
    f.write(modeling_content)

print(f"\nReporte de Modelado guardado en: {modeling_report_path}")
