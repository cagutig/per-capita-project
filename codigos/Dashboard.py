# %%
#!python.exe -m pip install --upgrade pip
#!python.exe -m pip install pandas
#!pip install jupyter-dash
#!pip install --upgrade jinja2 flask

# %%
import pandas as pd

df = pd.read_csv('../data/pib.csv', sep=',')
df.head()

# %%
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

maximo_ano = df['Year'].max()
df_ultimos_6_anos = df[df['Year'] >= maximo_ano - 5]
df_pivot_6_anos = df_ultimos_6_anos.pivot(index="Country Name", columns="Year", values="GDP per capita (current US$)").reset_index()

app = Dash(__name__)

background_image = "https://www.estrategia-sustentable.com.mx/wp-content/uploads/2022/06/pib.jpg"

app.layout = html.Div(style={
    'font-family': 'Roboto',
    'background-image': f'url({background_image})',
    'background-size': 'cover',
    'background-position': 'center',
    'padding': '20px'
}, children=[
    html.H1("Análisis de Indicadores Económicos y Sociales", style={'textAlign': 'center', 'color': '#FFCC00'}),

    html.Div([
        html.H2("Filtros", style={'textAlign': 'center', 'color': '#00274D'}),
        html.Div([
            html.Div([
                html.Label("Año de inicio", style={'font-weight': 'bold', 'color': '#00274D'}),
                dcc.Dropdown(
                    id="filtro_ano_inicio",
                    options=[{'label': ano, 'value': ano} for ano in sorted(df['Year'].unique(), reverse=True)],
                    placeholder="Selecciona año de inicio",
                    multi=False,
                    style={'width': '90%', 'padding': '5px'}
                )
            ], style={'width': '30%'}),

            html.Div([
                html.Label("Año de fin", style={'font-weight': 'bold', 'color': '#00274D'}),
                dcc.Dropdown(
                    id="filtro_ano_fin",
                    options=[{'label': ano, 'value': ano} for ano in sorted(df['Year'].unique(), reverse=True)],
                    placeholder="Selecciona año de fin",
                    multi=False,
                    style={'width': '90%', 'padding': '5px'}
                )
            ], style={'width': '30%', 'margin-left': '20px'}),

            html.Div([
                html.Label("Países", style={'font-weight': 'bold', 'color': '#00274D'}),
                dcc.Dropdown(
                    id="filtro_paises",
                    options=[{'label': pais, 'value': pais} for pais in df['Country Name'].unique()],
                    placeholder="Selecciona los países",
                    multi=True,
                    style={'width': '100%', 'padding': '5px', 'height': '100px'}
                )
            ], style={'width': '60%', 'margin-top': '10px'})
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'}),
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'border-radius': '10px', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Tendencia del PIB per Cápita", style={'textAlign': 'center', 'color': '#FFCC00'}),
        dcc.Graph(id="grafico_linea"),
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Tabla de PIB per Cápita - Últimos 6 Años", style={'textAlign': 'center', 'color': '#FFCC00'}),
        dash_table.DataTable(
            id='tabla_datos',
            columns=[{"name": str(i), "id": str(i)} for i in df_pivot_6_anos.columns],
            data=df_pivot_6_anos.to_dict('records'),
            page_size=10,
            style_table={'margin': 'auto', 'width': '80%'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': '#00274D',
                'fontWeight': 'bold',
                'color': '#FFCC00'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#E0E0E0'
                }
            ]
        )
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Mapa de Calor del PIB per Cápita (Último Año)", style={'textAlign': 'center', 'color': '#FFCC00'}),
        dcc.Graph(id="mapa_calor"),
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Análisis de Variables Económicas y Sociales", style={'textAlign': 'center', 'color': '#FFCC00'}),
        dcc.Dropdown(
            id="selector_variable",
            options=[
                {'label': 'Acceso a Electricidad (% de Población)', 'value': 'Access to electricity (% of population)'},
                {'label': 'Inversión Extranjera Directa (% del PIB)', 'value': 'Foreign direct investment, net inflows (% of GDP)'},
                {'label': 'Gasto en Educación (% del PIB)', 'value': 'Government expenditure on education, total (% of GDP)'},
                {'label': 'Crecimiento de la Población (%)', 'value': 'Population growth (annual %)'},
            ],
            value='Access to electricity (% of population)',
            style={'width': '50%', 'margin': 'auto', 'padding': '10px'}
        ),
        dcc.Graph(id="grafico_variable_economica")
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'})
])


@app.callback(
    Output("grafico_linea", "figure"),
    [Input("filtro_ano_inicio", "value"), Input("filtro_ano_fin", "value"), Input("filtro_paises", "value")]
)
def actualizar_grafico_linea(ano_inicio, ano_fin, paises_seleccionados):
    df_filtrado = df
    if paises_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Country Name'].isin(paises_seleccionados)]
    if ano_inicio and ano_fin:
        df_filtrado = df_filtrado[(df_filtrado['Year'] >= ano_inicio) & (df_filtrado['Year'] <= ano_fin)]
    fig = px.line(df_filtrado, x="Year", y="GDP per capita (current US$)", color="Country Name",
                  title="Tendencia del PIB per Cápita por País")
    fig.update_layout(xaxis_title="Año", yaxis_title="PIB per Cápita (US$)")
    return fig

@app.callback(
    Output("mapa_calor", "figure"),
    [Input("filtro_ano_fin", "value")]
)
def actualizar_mapa_calor(ano_seleccionado):
    ultimo_ano = df['Year'].max()
    df_filtrado = df[df['Year'] == (ano_seleccionado if ano_seleccionado else ultimo_ano)]
    limite_inferior = df_filtrado["GDP per capita (current US$)"].quantile(0.05)
    limite_superior = df_filtrado["GDP per capita (current US$)"].quantile(0.95)
    fig = px.choropleth(
        df_filtrado,
        locations="Country Code",
        color="GDP per capita (current US$)",
        color_continuous_scale="Viridis_r",
        hover_name="Country Name",
        title=f"Mapa de PIB per Cápita - {ano_seleccionado if ano_seleccionado else ultimo_ano}",
        range_color=(limite_inferior, limite_superior)
    )
    fig.update_layout(coloraxis_colorbar_title="PIB per Cápita (US$)")
    return fig

@app.callback(
    Output("grafico_variable_economica", "figure"),
    [Input("selector_variable", "value"), Input("filtro_ano_inicio", "value"), Input("filtro_ano_fin", "value"), Input("filtro_paises", "value")]
)
def actualizar_grafico_variable_economica(variable_seleccionada, ano_inicio, ano_fin, paises_seleccionados):
    df_filtrado = df
    if paises_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Country Name'].isin(paises_seleccionados)]
    if ano_inicio and ano_fin:
        df_filtrado = df_filtrado[(df_filtrado['Year'] >= ano_inicio) & (df_filtrado['Year'] <= ano_fin)]
    traducciones = {
        'Access to electricity (% of population)': 'Acceso a Electricidad (% de Población)',
        'Foreign direct investment, net inflows (% del PIB)': 'Inversión Extranjera Directa (% del PIB)',
        'Government expenditure on education, total (% of GDP)': 'Gasto en Educación (% del PIB)',
        'Population growth (annual %)': 'Crecimiento de la Población (%)'
    }
    titulo_variable = traducciones.get(variable_seleccionada, variable_seleccionada)
    fig = px.line(df_filtrado, x="Year", y=variable_seleccionada, color="Country Name",
                  title=f"Tendencia de {titulo_variable}")
    fig.update_layout(xaxis_title="Año", yaxis_title=titulo_variable)
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8069)


# %%



