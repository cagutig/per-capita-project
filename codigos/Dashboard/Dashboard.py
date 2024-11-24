# %%
#!python.exe -m pip install --upgrade pip
#!python.exe -m pip install pandas
#!pip install jupyter-dash
#!pip install --upgrade jinja2 flask

# %%
import pandas as pd

df = pd.read_csv('../../data/pib.csv', sep=',')
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

#background_image = "https://www.estrategia-sustentable.com.mx/wp-content/uploads/2022/06/pib.jpg"

app.layout = html.Div(style={   
    'font-family': 'Roboto',
    #'background-image': f'url({background_image})',
    'background-color': '#000000',
    'background-size': 'cover',
    'background-position': 'center',
    'padding': '20px'
}, children=[
    html.H1("Análisis de Indicadores Económicos y Sociales", style={'textAlign': 'center', 'color': '#FFCC00'}),

    html.Div([
    html.H2("Predicción", style={'textAlign': 'center', 'color': '#00274D', 'fontWeight': 'bold'}),

    # Primera fila: Años
    html.Div([
    html.Div([
        html.Label("Año de inicio", style={'font-weight': 'bold', 'color': '#00274D'}),
        dcc.Dropdown(
            id="filtro_ano_inicio",
            options=[{'label': ano, 'value': ano} for ano in sorted(df['Year'].unique())],
            placeholder="Año de inicio",
            multi=False,
            style={'width': '100%', 'padding': '5px'}
        )
    ], style={'width': '30%', 'padding': '5px'}),

    html.Div([
        html.Label("Año de fin", style={'font-weight': 'bold', 'color': '#00274D'}),
        dcc.Dropdown(
            id="filtro_ano_fin",
            options=[{'label': ano, 'value': ano} for ano in sorted(df['Year'].unique(), reverse=True)],
            placeholder="Año de fin",
            multi=False,
            style={'width': '100%', 'padding': '5px'}
        )
    ], style={'width': '30%', 'padding': '5px'}),

    html.Div([
        html.Label("Año de Predicción", style={'font-weight': 'bold', 'color': '#00274D'}),
        dcc.Input(
            id="ano_prediccion",
            type="text",
            value="2024",
            disabled=True,
            style={
                'padding': '10px',
                'backgroundColor': '#f0f0f0',
                'color': '#00274D',
                'border-radius': '5px',
                'border': '1px solid #ccc',
                'text-align': 'center',
                'font-weight': 'bold'
            }
        )
    ], style={'width': '30%', 'padding': '5px'})
], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'align-items': 'center'})
,

    # Segunda fila: País
    html.Div([
        html.Div([
            html.Label("País", style={'font-weight': 'bold', 'color': '#00274D'}),
            dcc.Dropdown(
                id="filtro_paises",
                options=[{'label': pais, 'value': pais} for pais in df['Country Name'].unique()],
                value="Colombia",
                multi=False,
                style={'width': '100%', 'padding': '5px'}
            )
        ], style={'width': '90%', 'padding': '5px'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

    # Tercera fila: Variables 1 y 2
    html.Div([
        html.Div([
            html.Label("Acceso a Electricidad (% de Población)", style={'font-weight': 'bold', 'color': '#00274D'}),
            dcc.Input(
                id="input_electricidad",
                type="number",
                placeholder="Valor",
                style={'width': '100%', 'padding': '5px'}
            )
        ], style={'width': '45%', 'padding': '5px'}),

        html.Div([
            html.Label("Inversión Extranjera (% del PIB)", style={'font-weight': 'bold', 'color': '#00274D'}),
            dcc.Input(
                id="input_inversion",
                type="number",
                placeholder="Valor",
                style={'width': '100%', 'padding': '5px'}
            )
        ], style={'width': '45%', 'padding': '5px'})
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'align-items': 'center'}),

    # Cuarta fila: Variables 3 y 4
    html.Div([
        html.Div([
            html.Label("Gasto en Educación (% del PIB)", style={'font-weight': 'bold', 'color': '#00274D'}),
            dcc.Input(
                id="input_educacion",
                type="number",
                placeholder="Valor",
                style={'width': '100%', 'padding': '5px'}
            )
        ], style={'width': '45%', 'padding': '5px'}),

        html.Div([
            html.Label("Crecimiento de la Población (%)", style={'font-weight': 'bold', 'color': '#00274D'}),
            dcc.Input(
                id="input_crecimiento",
                type="number",
                placeholder="Valor",
                style={'width': '100%', 'padding': '5px'}
            )
        ], style={'width': '45%', 'padding': '5px'})
    ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'align-items': 'center'})
    , 
    html.Div([
    html.Label("Predicción", style={'font-weight': 'bold', 'color': '#00274D'}),
    dcc.Input(
        id="campo_prediccion",
        type="number",
        value=None,
        disabled=True,
        style={
            'width': '90%', 'padding': '5px',
            'backgroundColor': '#f0f0f0',
            'color': '#00274D',
            'font-weight': 'bold',
             'justify-content': 'center', 'align-items': 'center','text-align': 'center',
        }
    )
], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'align-items': 'center'})
], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'border-radius': '10px', 'margin-bottom': '20px'})

    ,

    html.Div([
        html.H2("Tendencia del PIB per Cápita", style={ 'textAlign': 'center','color': '#00274D','fontWeight': 'bold'}),
        dcc.Graph(id="grafico_linea"),
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Análisis de Variables Económicas y Sociales", style={'textAlign': 'center','color': '#00274D','fontWeight': 'bold'}),
        dcc.Dropdown(
            id="selector_variable",
            options=[
                {'label': 'Acceso a Electricidad (% de Población)', 'value': 'Access to electricity (% of population)'},
                {'label': 'Inversión Extranjera Directa (% del PIB)', 'value': 'Foreign direct investment, net inflows (% of GDP)'},
                {'label': 'Gasto en Educación (% del PIB)', 'value': 'Government expenditure on education, total (% of GDP)'},
                {'label': 'Crecimiento de la Población (%)', 'value': 'Population growth (annual %)'},
            ],
            value='Population growth (annual %)',
            style={'width': '80%', 'margin': 'auto', 'padding': '10px'}
        ),
        dcc.Graph(id="grafico_variable_economica")
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Mapa de Calor del PIB per Cápita (Último Año)", style={'textAlign': 'center','color': '#00274D','fontWeight': 'bold'}),
        dcc.Graph(id="mapa_calor"),
    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}),

    html.Div([
        html.H2("Tabla de PIB per Cápita - Últimos 6 Años", style={'textAlign': 'center','color': '#00274D','fontWeight': 'bold'}),
        dash_table.DataTable(
            id='tabla_datos',
            columns=[
                {"name": "País", "id": "Country Name"},
                *[
                    {"name": str(ano), "id": str(ano), "type": "numeric", "format": {"specifier": ".1f"}}
                    for ano in df_pivot_6_anos.columns[1:]
                ]
            ],
            data=[],
            page_current=0,
            page_size=10,
            style_table={'margin': 'auto', 'width': '80%', 'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'minWidth': '80px', 
                'width': '80px',  
                'maxWidth': '80px',
                'whiteSpace': 'normal'
            },
            style_header={'backgroundColor': '#00274D', 'fontWeight': 'bold', 'color': '#FFCC00'},
            style_data_conditional=[]
        )


    ], style={'background-color': 'rgba(255, 255, 255, 0.8)', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'})
    ,

])

@app.callback(
    [
        Output("campo_prediccion", "value"),
        Output("input_electricidad", "value"),
        Output("input_inversion", "value"),
        Output("input_educacion", "value"),
        Output("input_crecimiento", "value")
    ],
    [Input("filtro_paises", "value")]
)
def actualizar_prediccion_y_valores(pais_seleccionado):
    prediccion = None
    electricidad = None
    inversion = None
    educacion = None
    crecimiento = None

    if pais_seleccionado:
        df_filtrado = df[df["Country Name"] == pais_seleccionado]
        if not df_filtrado.empty:
            maximo_ano_pais = df_filtrado["Year"].max()
            datos_max_ano = df_filtrado[df_filtrado["Year"] == maximo_ano_pais]

            def obtener_valor_o_promedio(columna):
                if columna in datos_max_ano and not datos_max_ano[columna].isnull().all():
                    return datos_max_ano[columna].values[0]
                elif columna in df_filtrado and not df_filtrado[columna].isnull().all():
                    return df_filtrado[columna].mean()
                return None

            electricidad = obtener_valor_o_promedio("Access to electricity (% of population)")
            inversion = obtener_valor_o_promedio("Foreign direct investment, net inflows (% of GDP)")
            educacion = obtener_valor_o_promedio("Government expenditure on education, total (% of GDP)")
            crecimiento = obtener_valor_o_promedio("Population growth (annual %)")

            if "GDP per capita (current US$)" in datos_max_ano and not datos_max_ano["GDP per capita (current US$)"].isnull().all():
                prediccion = datos_max_ano["GDP per capita (current US$)"].values[0]
            elif "GDP per capita (current US$)" in df_filtrado and not df_filtrado["GDP per capita (current US$)"].isnull().all():
                prediccion = df_filtrado["GDP per capita (current US$)"].mean()

    return prediccion, electricidad, inversion, educacion, crecimiento



@app.callback(
    Output("grafico_linea", "figure"),
    [Input("filtro_ano_inicio", "value"),
     Input("filtro_ano_fin", "value"),
     Input("filtro_paises", "value")]
)
def actualizar_grafico_linea(ano_inicio, ano_fin, pais_seleccionado):
    df_filtrado = df.copy()

    if ano_inicio is None:
        ano_inicio = df['Year'].min()
    if ano_fin is None:
        ano_fin = df['Year'].max()

    if pais_seleccionado:
        df_filtrado = df_filtrado[df_filtrado['Country Name'] == pais_seleccionado]

    df_filtrado = df_filtrado[(df_filtrado['Year'] >= ano_inicio) & (df_filtrado['Year'] <= ano_fin)]

    if df_filtrado.empty:
        return px.line(title="No hay datos disponibles para los filtros seleccionados")

    df_filtrado = df_filtrado.dropna(subset=["Year", "GDP per capita (current US$)"])

    fig = px.line(
        df_filtrado,
        x="Year",
        y="GDP per capita (current US$)",
        color="Country Name",
        title="Tendencia del PIB per Cápita por País"
    )

    if not df_filtrado.empty:
        ultimo_valor = df_filtrado.groupby("Country Name")["GDP per capita (current US$)"].last().reset_index()
        ultimo_valor["Year"] = df["Year"].max() + 1 
        fig.add_scatter(
            x=ultimo_valor["Year"],
            y=ultimo_valor["GDP per capita (current US$)"],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            name="PIB Predicho",
            text=ultimo_valor["Country Name"],
            hoverinfo="text+y"
        )

    fig.update_layout(xaxis_title="Año", yaxis_title="PIB per Cápita (US$)")

    return fig




@app.callback(
    Output("mapa_calor", "figure"),
    [Input("filtro_ano_fin", "value"), Input("filtro_paises", "value")]
)
def actualizar_mapa_calor(ano_seleccionado, pais_seleccionado):
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

    if pais_seleccionado:
        pais_datos = df_filtrado[df_filtrado["Country Name"] == pais_seleccionado]
        if not pais_datos.empty:
            fig.add_scattergeo(
                locations=pais_datos["Country Code"],
                locationmode="ISO-3",
                marker=dict(
                    size=15,
                    symbol="triangle-down", 
                    color="red",
                    line=dict(width=2, color="black")
                ),
                name="País seleccionado",
                hoverinfo="skip"
            )
    
    fig.update_layout(coloraxis_colorbar_title="PIB per Cápita (US$)")

    return fig



@app.callback(
    [Output("tabla_datos", "data"),
     Output("tabla_datos", "page_current"),
     Output("tabla_datos", "style_data_conditional")],
    [Input("filtro_paises", "value"), Input("filtro_ano_fin", "value")]
)
def actualizar_tabla(pais_seleccionado, ano_seleccionado):
    df_ultimos_6_anos = df[df['Year'] >= maximo_ano - 5]
    df_pivot_6_anos = df_ultimos_6_anos.pivot(index="Country Name", columns="Year", values="GDP per capita (current US$)").reset_index()

    data = df_pivot_6_anos.to_dict('records')
    page_current = 0
    style_data_conditional = [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#E0E0E0'
        }
    ]

    if pais_seleccionado:
        index_pais = df_pivot_6_anos[df_pivot_6_anos["Country Name"] == pais_seleccionado].index
        if not index_pais.empty:
            page_current = index_pais[0] // 10
            style_data_conditional.append({
                'if': {'filter_query': f'{{Country Name}} = "{pais_seleccionado}"'},
                'backgroundColor': '#FFCC00',
                'color': '#00274D',
                'fontWeight': 'bold'
            })

    return data, page_current, style_data_conditional



@app.callback(
    Output("grafico_variable_economica", "figure"),
    [Input("selector_variable", "value"), Input("filtro_ano_inicio", "value"), Input("filtro_ano_fin", "value"), Input("filtro_paises", "value")]
)
def actualizar_grafico_variable_economica(variable_seleccionada, ano_inicio, ano_fin, paises_seleccionados):
    df_filtrado = df
    if paises_seleccionados:
        if isinstance(paises_seleccionados, str):
            df_filtrado = df_filtrado[df_filtrado['Country Name'] == paises_seleccionados]
        else:
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
#    app.run_server(debug=True, host='0.0.0.0', port=8069) 
    app.run_server(debug=True, port=8069) 


# %%



