import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# --- 1. Configuración y Carga de Datos ---
# El archivo siniestros_viales_limpios.csv contiene los datos pre-procesados
DATA_PATH = 'siniestros_viales_limpios.csv'
df = pd.read_csv(DATA_PATH)

# Asegurar que las columnas de fecha/hora sean del tipo correcto
df['fecha_hora_accidente'] = pd.to_datetime(df['fecha_hora_accidente'])
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'], ordered=True)

# Mapeo de Localidades (ejemplo simplificado, se asume que el código es el nombre)
localidad_map = {
    1: 'Usaquén', 2: 'Chapinero', 3: 'Santa Fe', 4: 'San Cristóbal', 5: 'Usme',
    6: 'Tunjuelito', 7: 'Bosa', 8: 'Kennedy', 9: 'Fontibón', 10: 'Engativá',
    11: 'Suba', 12: 'Barrios Unidos', 13: 'Teusaquillo', 14: 'Los Mártires',
    15: 'Antonio Nariño', 16: 'Puente Aranda', 17: 'La Candelaria', 18: 'Rafael Uribe Uribe',
    19: 'Ciudad Bolívar', 20: 'Sumapaz'
}
df['nombre_localidad'] = df['codigo_localidad'].map(localidad_map).fillna('Desconocida')

# --- 2. Inicialización de la Aplicación Dash ---
app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# --- 3. Definición del Layout del Dashboard ---
app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'padding': '20px'}, children=[
    html.H1("Dashboard de Analítica Predictiva de Siniestralidad Vial en Bogotá", 
            style={'textAlign': 'center', 'color': '#343a40'}),
    
    html.Div([
        html.Div([
            html.Label("Seleccionar Localidad:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='localidad-dropdown',
                options=[{'label': loc, 'value': loc} for loc in sorted(df['nombre_localidad'].unique())],
                value=sorted(df['nombre_localidad'].unique()),
                multi=True
            ),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Seleccionar Gravedad:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='gravedad-dropdown',
                options=[
                    {'label': 'Alta Gravedad (Lesionado/Muerto)', 'value': 1},
                    {'label': 'Baja Gravedad (Solo Daños)', 'value': 0}
                ],
                value=[0, 1],
                multi=True
            ),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'marginBottom': '20px', 'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '5px'}),

    # Fila 1: Análisis Temporal
    html.Div([
        html.Div([
            html.H3("1. Mapa de Calor Temporal (Hora vs. Día)", style={'textAlign': 'center'}),
            dcc.Graph(id='heatmap-temporal'),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey'}),
        
        html.Div([
            html.H3("2. Distribución de Gravedad", style={'textAlign': 'center'}),
            dcc.Graph(id='gravedad-distribucion'),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

    # Fila 2: Análisis Geográfico y Métricas
    html.Div([
        html.Div([
            html.H3("3. Siniestros por Localidad", style={'textAlign': 'center'}),
            dcc.Graph(id='localidad-bar-chart'),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey'}),
        
        html.Div([
            html.H3("4. Métricas Clave del Modelo Predictivo", style={'textAlign': 'center'}),
            html.Div([
                html.P(f"ROC AUC Score: 0.76", style={'fontSize': '1.2em', 'fontWeight': 'bold', 'color': '#007bff'}),
                html.P(f"Recall (Alta Gravedad): 0.88", style={'fontSize': '1.2em', 'fontWeight': 'bold', 'color': '#28a745'}),
                html.P("El modelo tiene una capacidad de discriminación moderada a buena.", style={'fontSize': '0.9em', 'color': '#6c757d'}),
                html.Img(src=app.get_asset_url('roc_curve.png'), style={'width': '100%', 'marginTop': '10px'})
            ], style={'padding': '20px', 'textAlign': 'left'}),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
])

# --- 4. Callbacks para la Interactividad ---

@app.callback(
    [Output('heatmap-temporal', 'figure'),
     Output('gravedad-distribucion', 'figure'),
     Output('localidad-bar-chart', 'figure')],
    [Input('localidad-dropdown', 'value'),
     Input('gravedad-dropdown', 'value')]
)
def update_graphs(selected_localidades, selected_gravedad):
    # Filtrar datos
    filtered_df = df[
        df['nombre_localidad'].isin(selected_localidades) & 
        df['gravedad_binaria'].isin(selected_gravedad)
    ]

    # --- Gráfico 1: Mapa de Calor Temporal ---
    # Asegurar que los datos para el heatmap sean numéricos y no tengan NaNs
    heatmap_data = filtered_df.groupby(['dia_semana', 'hora_del_dia']).size().reset_index(name='conteo')
    heatmap_data['conteo'] = heatmap_data['conteo'].fillna(0).astype(int)
    
    # Asegurar que el eje Y (días) esté en el orden correcto
    heatmap_data['dia_semana'] = pd.Categorical(heatmap_data['dia_semana'], categories=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'], ordered=True)
    heatmap_data = heatmap_data.sort_values('dia_semana')

    fig_heatmap = px.density_heatmap(
        heatmap_data, 
        x="hora_del_dia", 
        y="dia_semana", 
        z="conteo", 
        title="Concentración de Siniestros por Hora y Día",
        color_continuous_scale="Viridis",
        labels={'hora_del_dia': 'Hora del Día (0-23)', 'dia_semana': 'Día de la Semana', 'conteo': 'Conteo de Siniestros'}
    )
    fig_heatmap.update_layout(xaxis={'dtick': 1}) # Mostrar todas las horas

    # --- Gráfico 2: Distribución de Gravedad ---
    gravedad_counts = filtered_df['gravedad_binaria'].map({1: 'Alta Gravedad', 0: 'Baja Gravedad'}).value_counts().reset_index(name='Conteo')
    gravedad_counts.columns = ['Gravedad', 'Conteo']
    
    fig_gravedad = px.pie(
        gravedad_counts, 
        names='Gravedad', 
        values='Conteo', 
        title='Distribución de Gravedad de Siniestros',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # --- Gráfico 3: Siniestros por Localidad ---
    localidad_counts = filtered_df['nombre_localidad'].value_counts().reset_index(name='Conteo')
    localidad_counts.columns = ['Localidad', 'Conteo']
    localidad_counts = localidad_counts.sort_values('Conteo', ascending=False).head(10) # Top 10

    fig_localidad = px.bar(
        localidad_counts, 
        x='Localidad', 
        y='Conteo', 
        title='Top 10 Localidades con Mayor Siniestralidad',
        color='Conteo',
        color_continuous_scale=px.colors.sequential.Reds
    )
    fig_localidad.update_layout(xaxis={'categoryorder':'total descending'})

    return fig_heatmap, fig_gravedad, fig_localidad

# --- 5. Ejecución del Servidor ---
if __name__ == '__main__':
    # El modo debug permite recargar automáticamente al guardar cambios
    # El puerto 8050 es el puerto estándar de Dash
    app.run(debug=True, port=8050)
