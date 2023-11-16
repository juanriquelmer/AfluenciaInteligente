import json
import requests
from pytz import timezone
from datetime import datetime
from Counter import Runner  # Importa la función desde Proyecto.py

# link pagina web:
# https://main.d7a6ikqkx1vx5.amplifyapp.com/

###############################################################################
# Model
model = 'yolov8x.pt'
modo = 'ROI'
confidence = 0.2
jpg_path = r"/home/jose/Git/AfluenciaInteligente/detector_module/Tarea_imgs/IMG_1314.JPG"

###############################################################################
# Web
santiago_timezone = timezone('Chile/Continental')
now = datetime.now(santiago_timezone).strftime("%Y-%m-%d %H:%M:%S")
dia = datetime.now(santiago_timezone).strftime("%A")

zona = 1

# Obtiene el valor de cantidad usando la función obtener_cantidad del script proyecto.py
cantidad = Runner(jpg_path, modo, confidence, model)
# Define la URL de la API Gateway
api_url = "https://dqrqv2q9jg.execute-api.sa-east-1.amazonaws.com/deploy"  # Reemplaza con la URL de tu API Gateway

# Datos a enviar en la solicitud POST
data = {
    "zona": str(zona),
    "cantidad": cantidad,
    "tiempo": now,
    "dia": dia
}
# Convierte los datos a JSON
data_json = json.dumps(data)

# Realiza la solicitud POST
response = requests.post(api_url, data=data_json, headers={"Content-Type": "application/json"})

# Verifica la respuesta
if response.status_code == 200:
    print(f"Solicitud exitosa! Respuesta: {response.text}")
    print("Resultados publicados en:", "https://main.d7a6ikqkx1vx5.amplifyapp.com/")
else:
    print(f"Error en la solicitud: {response.status_code} - {response.text}")
