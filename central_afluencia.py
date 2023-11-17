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
periodo_captura= 20

# Model
model = 'yolov8x.pt'
modo = 'normal'
confidence = 0.2

# Definicón de la URL de la API
api_url = "https://dqrqv2q9jg.execute-api.sa-east-1.amazonaws.com/deploy" 

# Obtenemos la zona horaria de Santiago de Chile
santiago_timezone = timezone('Chile/Continental')

zona = 1 

def run_cam_module(module):
    module.capturing = True
    try:
        module.capture() 
    except KeyboardInterrupt:
        print("Captura interrumpida.")
    finally:
        module.capturing = False
        print("Captura detenida.")

def main():
    cam_module = CameraModule(capture_period=periodo_captura)
    detector_photo_count = 0 
    thread_cam = threading.Thread(target = run_cam_module, args=(cam_module,))
    thread_cam.start()

    while True:

        sleep(periodo_captura + 1)
        jpg_path = cam_module._get_last_photo_path()
        #jpg_path = cam_module.get_last_photo_path(detector_photo_count)

        cantidad = Runner(jpg_path, modo, confidence, model)
        
        # Obtén la hora actual en Santiago de Chile automáticamente
        now = datetime.now(santiago_timezone).strftime("%Y-%m-%d %H:%M:%S")
        dia = datetime.now(santiago_timezone).strftime("%A")
        
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
        
        """
        if cam_module.get_actual_count() > detector_photo_count
            count = Runner(jpg_path, modo, confidence, model)
            #detector_photo_count + 1
        else:
            pass
        """
        
    #runner(path)


if __name__ == "__main__":
    main()
    