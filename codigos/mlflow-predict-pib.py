# Paso 1: Importar Librerías y Preparar los Datos
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
import mlflow.sklearn

# Leer el archivo CSV desde el repositorio de GitHub
url = 'https://raw.githubusercontent.com/cagutig/per-capita-project/refs/heads/main/data/df_gdp_concatenado.csv'
df = pd.read_csv(url)

# Organizar y preparar los datos
df = df.sort_values(['Country Name', 'Year'])
df['Original Country Name'] = df['Country Name']

# Paso 2: Crear Características con Retardos (Lags)
for col in ['GDP per capita (current US$)', 'Access to electricity (% of population)',
            'Foreign direct investment, net inflows (% of GDP)', 
            'Government expenditure on education, total (% of GDP)',
            'Population growth (annual %)']:
    df[f'{col}_lag1'] = df.groupby('Country Name')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('Country Name')[col].shift(2)

df = df.dropna(subset=[f'{col}_lag1' for col in ['GDP per capita (current US$)', 
                                                 'Access to electricity (% of population)',
                                                 'Foreign direct investment, net inflows (% of GDP)', 
                                                 'Government expenditure on education, total (% of GDP)',
                                                 'Population growth (annual %)']])

# Paso 3: One-Hot Encoding para la Variable de País
df = pd.get_dummies(df, columns=['Country Name'], drop_first=True)

# Paso 4: Dividir los Datos en Entrenamiento y Validación (80/20)
X = df[[f'{col}_lag1' for col in ['GDP per capita (current US$)', 
                                  'Access to electricity (% of population)']] + 
       [f'{col}_lag2' for col in ['GDP per capita (current US$)', 
                                  'Access to electricity (% of population)']] +
       [col for col in df.columns if 'Country Name_' in col]]  # Añadimos las columnas del encoding
y = df['GDP per capita (current US$)']

# División en 80% entrenamiento y 20% validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 5: Imputar Valores Faltantes
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Iniciar MLflow
mlflow.set_experiment("GDP_Prediction_Model")

with mlflow.start_run():
    # Paso 6: Entrenar el Modelo Ajustado con RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200,         # Número de árboles en el bosque
        max_depth=15,             # Profundidad máxima de cada árbol
        min_samples_split=5,      # Muestras mínimas necesarias para dividir un nodo
        min_samples_leaf=4,       # Muestras mínimas necesarias en una hoja
        random_state=42
    )

    # Entrenar el modelo ajustado
    model.fit(X_train_imputed, y_train)

    # Realice predicciones de validación
    y_val_pred = model.predict(X_val_imputed)
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)

    # Registrar los parámetros del modelo
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 4)

    # Registrar el modelo
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Registrar las métricas
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)

    # Imprimir las métricas para verificar
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Paso 7: Realizar la Proyección para 2024
test_data = df[df['Year'] == 2023].copy()
test_data['Year'] = 2024  # Cambiar el año a 2024 para reflejar la predicción

X_test = test_data[[f'{col}_lag1' for col in ['GDP per capita (current US$)', 
                                              'Access to electricity (% of population)']] + 
                   [f'{col}_lag2' for col in ['GDP per capita (current US$)', 
                                              'Access to electricity (% of population)']] +
                   [col for col in df.columns if 'Country Name_' in col]]

X_test_imputed = imputer.transform(X_test)

# Realizar la proyección para todas las variables de interés
test_data['GDP per capita (current US$)'] = model.predict(X_test_imputed)  # Predicción del PIB

# Seleccionar las columnas deseadas para la salida final
columns_output = ['Original Country Name', 'Country Code', 'Year', 
                  'Access to electricity (% of population)',
                  'Foreign direct investment, net inflows (% of GDP)',
                  'GDP per capita (current US$)',
                  'Government expenditure on education, total (% of GDP)',
                  'Population growth (annual %)']

output_data = test_data[columns_output]

# Mostrar las primeras filas de la proyección
print(output_data.head())
