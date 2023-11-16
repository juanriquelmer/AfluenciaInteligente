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

        count = Runner(jpg_path, modo, confidence, model)
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
    