from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import joblib
import uvicorn


# Inicializar FastAPI
app = FastAPI()

# Clase para validar los datos de entrada
class PredictionInput(BaseModel):
    Access_to_electricity: float
    Foreign_direct_investment: float
    Government_expenditure: float
    Population_growth: float
    Country_Name: str

# Leer los datos y entrenar el modelo solo una vez al iniciar el servidor
url = 'https://raw.githubusercontent.com/cagutig/per-capita-project/refs/heads/main/data/df_gdp_concatenado.csv'
df = pd.read_csv(url)

# Preprocesamiento de datos y entrenamiento (resumido para claridad)
columnas_a_imputar = [
    'Access to electricity (% of population)',
    'Foreign direct investment, net inflows (% of GDP)',
    'Government expenditure on education, total (% of GDP)',
    'Population growth (annual %)'
]

# Imputar datos faltantes
def imputar_knn_columnas_seleccionadas(grupo):
    columnas_presentes = [col for col in columnas_a_imputar if col in grupo.columns]
    if len(columnas_presentes) == 0:
        return grupo
    n_neighbors = min(3, len(grupo))
    imputer = KNNImputer(n_neighbors=n_neighbors)
    if grupo[columnas_presentes].isnull().all(axis=0).any():
        return grupo
    imputado = pd.DataFrame(
        imputer.fit_transform(grupo[columnas_presentes]),
        columns=columnas_presentes,
        index=grupo.index
    )
    for col in columnas_presentes:
        grupo[col] = imputado[col]
    return grupo

df_imputado = df.groupby('Country Name', group_keys=False).apply(imputar_knn_columnas_seleccionadas)

# Crear características con retardos (lags)
for col in columnas_a_imputar + ['GDP per capita (current US$)']:
    df_imputado[f'{col}_lag1'] = df_imputado.groupby('Country Name')[col].shift(1)
    df_imputado[f'{col}_lag2'] = df_imputado.groupby('Country Name')[col].shift(2)

# Borrar filas con valores nulos en las columnas lag
columnas_lag = [f'{col}_lag1' for col in columnas_a_imputar] + [f'{col}_lag2' for col in columnas_a_imputar]
df_imputado = df_imputado.dropna(subset=columnas_lag)

# Cargar el modelo y las columnas
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Endpoint para hacer predicciones
@app.post("/predict/")
def predict(input_data: PredictionInput):

    pais_pred = input_data.Country_Name
    datos_prediccion = df_imputado[df_imputado['Country Name'] == pais_pred]
    datos_prediccion = datos_prediccion[datos_prediccion['Year'] == datos_prediccion['Year'].max()]


    # Crear DataFrame con los valores enviados por el usuario
    input_df = pd.DataFrame([{
        "Country Name": input_data.Country_Name,
        "Year": 2024,
        "Access to electricity (% of population)": input_data.Access_to_electricity,
        "Foreign direct investment, net inflows (% of GDP)": input_data.Foreign_direct_investment,
        "Government expenditure on education, total (% of GDP)": input_data.Government_expenditure,
        "Population growth (annual %)": input_data.Population_growth
    }])

    # Crear columnas lag
    input_df[f'Access to electricity (% of population)_lag1'] = datos_prediccion['Access to electricity (% of population)'].iloc[0]
    input_df[f'Foreign direct investment, net inflows (% of GDP)_lag1'] = datos_prediccion['Foreign direct investment, net inflows (% of GDP)'].iloc[0]
    input_df[f'Government expenditure on education, total (% of GDP)_lag1'] = datos_prediccion['Government expenditure on education, total (% of GDP)'].iloc[0]
    input_df[f'Population growth (annual %)_lag1'] = datos_prediccion['Population growth (annual %)'].iloc[0]
    input_df[f'GDP per capita (current US$)_lag1'] = datos_prediccion['GDP per capita (current US$)'].iloc[0]
    
    input_df[f'Access to electricity (% of population)_lag2'] = datos_prediccion['Access to electricity (% of population)_lag1'].iloc[0]
    input_df[f'Foreign direct investment, net inflows (% of GDP)_lag2'] = datos_prediccion['Foreign direct investment, net inflows (% of GDP)_lag1'].iloc[0]
    input_df[f'Government expenditure on education, total (% of GDP)_lag2'] = datos_prediccion['Government expenditure on education, total (% of GDP)_lag1'].iloc[0]
    input_df[f'Population growth (annual %)_lag2'] = datos_prediccion['Population growth (annual %)_lag1'].iloc[0]
    input_df[f'GDP per capita (current US$)_lag2'] = datos_prediccion['GDP per capita (current US$)_lag1'].iloc[0]
    
    # One-Hot Encoding para 'Country Name'
    input_df = pd.get_dummies(input_df, columns=['Country Name'], drop_first=True)


    # Agregar columnas faltantes para que coincidan con las del modelo
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]
    
    # Hacer la predicción
    prediction = model.predict(input_df)[0]
    return {"prediction": prediction}


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
