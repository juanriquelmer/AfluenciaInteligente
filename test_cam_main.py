from camera_module.camera_module import CameraModule


def main():
    cam_module = CameraModule()

    cam_module.capturing = True
    try:
        cam_module.capture()
    except KeyboardInterrupt:
        print("Captura interrumpida.")
    finally:
        cam_module.capturing = False
        print("Captura detenida.")

if __name__ == "__main__":
    main()