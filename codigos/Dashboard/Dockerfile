# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos del directorio actual al contenedor
COPY . /app

# Instala las dependencias necesarias
RUN pip install --no-cache-dir dash plotly pandas jinja2 flask requests

# Expone el puerto del tablero
EXPOSE 8069

# Comando para ejecutar la aplicación
CMD ["python", "Dashboard.py"]
