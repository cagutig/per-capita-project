import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.impute import KNNImputer


# Leer el archivo CSV desde el repositorio de GitHub
url = 'https://raw.githubusercontent.com/cagutig/per-capita-project/refs/heads/main/data/df_gdp_concatenado.csv'
df = pd.read_csv(url)

# Definir el servidor para llevar el registro de modelos y artefactos
# mlflow.set_tracking_uri('http://0.0.0.0:5000')
# Registrar el experimento
mlflow.set_experiment("GDP_Prediction_Model")

# Organizar y preparar los datos
df = df.sort_values(['Country Name', 'Year'])

# Imputar datos faltantes en las columna dadas
# Crear una copia del DataFrame para no modificar el original
df_imputado = df.copy()

# Definir las columnas a imputar
columnas_a_imputar = ['Access to electricity (% of population)', 
                      'Foreign direct investment, net inflows (% of GDP)', 
                      'Government expenditure on education, total (% of GDP)',
                      'Population growth (annual %)']

# Función para imputar columnas seleccionadas
def imputar_knn_columnas_seleccionadas(grupo):
    # Seleccionar las columnas presentes en el grupo
    columnas_presentes = [col for col in columnas_a_imputar if col in grupo.columns]
    
    # Si no hay columnas presentes para imputar, devolver el grupo sin cambios
    if len(columnas_presentes) == 0:
        return grupo

    # Ajustar el número de vecinos dinámicamente según el tamaño del grupo
    n_neighbors = min(3, len(grupo))
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Verificar si todas las columnas seleccionadas tienen al menos un valor no nulo
    if grupo[columnas_presentes].isnull().all(axis=0).any():
        print(f"Saltando grupo {grupo['Country Name'].iloc[0]}: Todas las columnas están vacías.")
        return grupo
    
    # Imputar solo las columnas presentes
    imputado = pd.DataFrame(
        imputer.fit_transform(grupo[columnas_presentes]),
        columns=columnas_presentes,
        index=grupo.index
    )
    
    # Actualizar únicamente las columnas seleccionadas
    for col in columnas_presentes:
        grupo[col] = imputado[col]
    
    return grupo

# Agrupar por 'Country Name' y aplicar la imputación en cada grupo
df_imputado = df_imputado.groupby('Country Name', group_keys=False).apply(imputar_knn_columnas_seleccionadas)

# Crear Características con Retardos (Lags)
for col in ['GDP per capita (current US$)', 'Access to electricity (% of population)',
            'Foreign direct investment, net inflows (% of GDP)', 
            'Government expenditure on education, total (% of GDP)',
            'Population growth (annual %)']:
    df_imputado[f'{col}_lag1'] = df_imputado.groupby('Country Name')[col].shift(1)
    df_imputado[f'{col}_lag2'] = df_imputado.groupby('Country Name')[col].shift(2)

# Borrar filas con valores nulos en las columnas lag
columnas_lag = [f'{col}_lag1' for col in columnas_a_imputar] + [f'{col}_lag2' for col in columnas_a_imputar]
df_imputado = df_imputado.dropna(subset=columnas_lag)

# One-Hot Encoding para la Variable de País
df = pd.get_dummies(df_imputado, columns=['Country Name'], drop_first=True)

# Dividir el set de datos en entrenamiento y prueba

# Definir la variable dependiente (lo que queremos predecir)
y = df['GDP per capita (current US$)']

# Definir las variables independientes (todas las demás columnas relevantes)
X = df.drop(['GDP per capita (current US$)', 'Country Code'], axis=1)

# División en 80% entrenamiento y 20% validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar MLflow
mlflow.set_experiment("GDP_Prediction_Model")

with mlflow.start_run():
    # Entrenar el Modelo Ajustado con RandomForestRegressor
    n_estimators = 200
    max_depth = 15
    min_samples_split = 5
    min_samples_leaf = 4

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Entrenar el modelo ajustado
    model.fit(X_train, y_train)

    # Realizar predicciones de validación
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)

    # Registrar los parámetros del modelo
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)

    # Registrar el modelo
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Registrar las métricas
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)

    # Imprimir las métricas para verificar
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Realizar la Proyección para 2024, teniendo en cuenta los datos de 2023
# Filtrar los datos del año 2023
data_2023 = df_imputado[df_imputado['Year'] == 2023]
data_2024 = data_2023.copy()

#Crear datos para 2024
data_2024['Year'] = 2024  # Cambiar el año a 2024 para reflejar la predicción
data_2024

# Generar las características con retardos para 2024
for col in ['GDP per capita (current US$)', 'Access to electricity (% of population)',
            'Foreign direct investment, net inflows (% of GDP)', 
            'Government expenditure on education, total (% of GDP)',
            'Population growth (annual %)']:
    data_2024[f'{col}_lag1'] = data_2023[col]
    data_2024[f'{col}_lag2'] = data_2023[f'{col}_lag1']

# Mantener solo las columnas necesarias
columnas_necesarias = columnas_lag + ['Year']  # Retardos y el año

# Aplicar One-Hot Encoding para la columna 'Country Name'
data_2024 = pd.get_dummies(data_2024, columns=['Country Name'], drop_first=True)

# Asegurar que las columnas coincidan con las del modelo entrenado
missing_cols = set(X_train.columns) - set(data_2024.columns)
for col in missing_cols:
    data_2024[col] = 0  # Añadir columnas faltantes con valores 0

# Ordenar las columnas para que coincidan con las de entrenamiento
data_2024 = data_2024[X_train.columns]

# Realizar las predicciones para 2024
data_2024['GDP per capita (current US$)_predicted'] = model.predict(data_2024)

# Añadir la columna 'Country Name' de nuevo para claridad
data_2024['Country Name'] = data_2023['Country Name'].values

# Mostrar las predicciones por país
predicciones_2024 = data_2024[['Country Name', 'Year', 'GDP per capita (current US$)_predicted']]

# Mostrar las primeras filas de la proyección
print(predicciones_2024.head())

