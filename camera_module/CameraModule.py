import os
import time
import cv2


class CameraModule:
    """
    Clase para gestionar la cámara y su funcionalidad de captura.
    """
    def __init__(self, camera_index=0):
        self.capturing = False  # Flag para saber si está capturando o no
        self.photo_count = 0  # Contador de fotos tomadas
        self.max_photos = 3  # Máximo de fotos que se pueden tomar
        self.photo_format = ".jpg"
        self.photo_directory = "CameraModule_Log"
        self.camera_index = camera_index  # Índice de la cámara
        self.capture_period = 10  # Tiempo entre capturas

        # Crear directorio para las fotos si no existe
        if not os.path.exists(self.photo_directory):
            os.makedirs(self.photo_directory)

    def _get_photo_path(self, photo_number):
        """
        Genera y retorna la ruta completa de la foto basada en el número proporcionado.
        """
        photo_name = f"{photo_number:03d}{self.photo_format}"
        return os.path.join(self.photo_directory, photo_name)

    def save_photo(self, photo_number):
        """
        Captura y guarda una foto.
        """
        photo_path = self._get_photo_path(photo_number)
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(photo_path, frame)
            print(f"Foto {photo_path} guardada.")
        else:
            print("Error capturing photo.")
            self._handle_error()

    def delete_photo(self, photo_number):
        """
        Elimina una foto basada en el número proporcionado.
        """
        photo_path = self._get_photo_path(photo_number)
        if os.path.exists(photo_path):
            os.remove(photo_path)
            print(f"Foto {photo_path} eliminada.")
        else:
            print("Error eliminando la foto.")
            self._handle_error()

    def capture(self):
        """
        Comienza la captura de fotos en intervalos definidos por capture_period.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Error abriendo la cámara.")
            self._handle_error()
            return

        while self.capturing:
            self.save_photo(self.photo_count)
            self.photo_count = (self.photo_count + 1) % self.max_photos
            time.sleep(self.capture_period)

        self.cap.release()

    def _handle_error(self):
        """
        Maneja los errores que pueden surgir durante la captura o gestión de fotos.
        """
        pass
    