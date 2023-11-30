from camera_module import CameraModule
import json
import requests
from pytz import timezone
from datetime import datetime
from Counter import Runner
import threading
from time import sleep

## ARGS ##

# Camera
numero_fotos_inicial = 10
periodo_captura= 5
max_retries = 3
retry_delay = 1

# Model
model = 'yolov8x.pt'
modo = 'normal'
confidence = 0.2

# Definicón de la URL de la API
api_url = "https://dqrqv2q9jg.execute-api.sa-east-1.amazonaws.com/deploy" 

# Obtenemos la zona horaria de Santiago de Chile
santiago_timezone = timezone('Chile/Continental')

zona = 1 

def send_data(cantidad, flag=0):
    """Envía los datos a la API."""

    tiempo = datetime.now(santiago_timezone).strftime("%Y-%m-%d %H:%M:%S")
    dia = datetime.now(santiago_timezone).strftime("%A")
    
    # Datos a enviar en la solicitud POST
    data = {
        "zona": str(zona),
        "cantidad": cantidad,
        "tiempo": tiempo,
        "dia": dia,
        "flag": flag
    }
   
    # Convierte los datos a un formato que se pueda enviar
    data_json = json.dumps(data)

    # Envía la solicitud POST
    response = requests.post(api_url, data=data_json, headers={"Content-Type": "application/json"})
    
    return data

finish_flag = False
def run_cam_module(module):
    """Función que corre el módulo de la cámara."""
    global finish_flag

    module.capturing = True
    try:
        module.capture(max_retries=max_retries, retry_delay=retry_delay)
    except KeyboardInterrupt:
        print("Captura interrumpida.")
        module.close_camera()
    finally:
        module.close_camera()
        
def main():
    """Función principal del programa."""

    # Inicializar el módulo de la cámara
    cam_module = CameraModule(capture_period=periodo_captura)
    
    start_error = cam_module.initialize(numero_fotos_inicial = numero_fotos_inicial,initial_photo_period=0.5)
    if start_error:
        print ("-----------------------------\n")
        print("Error inicializando el módulo de la cámara.")
        print ("-----------------------------\n")
        return

    last_photo_path = cam_module._generate_last_photo_path() # Esta es la ultima foto de la inicialización

    # Iniciar el thread de la cámara
    thread_cam = threading.Thread(target = run_cam_module, args=(cam_module,))
    thread_cam.start()
    
    error = 0 # contador de errores

    # loop principal para corre el modelo y enviar los datos
    capturing = cam_module.get_capturing()

    try:
        while capturing:
            sleep(periodo_captura + 1)
            jpg_path = cam_module._generate_last_photo_path()
            
            # En caso que no haya una foto nueva, se salta el resto del loop
            # y en caso de que se repita la misma foto 3 veces, se envía un flag
            # para decir que se esta fuera de servicio
            if jpg_path == last_photo_path:
                error += 1
                if error < 3:
                    print("-----------------------------\n")
                    print("No hay foto nueva a ser analizada, se repite:", jpg_path)
                    print("-----------------------------\n")

                    continue  # Se parte el loop desde el principio
                else:
                    print(send_data(0, flag=1))

                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    print("Se ha enviado un flag de fuera de servicio.")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    error = 0 

            last_photo_path = jpg_path

            # Correr el modelo
            cantidad = Runner(jpg_path, modo, confidence, model)

            # Enviar los datos a la API
            print("------Datos enviados------")
            print(send_data(cantidad))
            print("--------------------------\n")

            capturing = cam_module.get_capturing()
            print("Capturando: ", capturing)
    except KeyboardInterrupt:
        print("Captura interrumpida.")
        cam_module.close_camera()

if __name__ == "__main__":
    main()
    