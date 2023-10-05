import os
import time
import cv2
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import numpy as np


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
    
    def _calculate_metrics(self, image_np):
        # Varianza del histograma de luminosidad
        variance = np.var(image_np)
        
        # Entropía de la imagen
        hist = cv2.calcHist([image_np], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = stats.entropy(hist)
        
        return variance, entropy

    def _mse(self, image_a, image_b):
        # Calcular el error cuadrático medio entre dos imágenes
        err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
        err /= float(image_a.shape[0] * image_a.shape[1])
        return err
    
    def _compare_images(self, image_np, reference_image_np):
        s = ssim(image_np, reference_image_np)
        e = self._mse(image_np, reference_image_np)
        return s, e

    def _process_image(self, image_path):
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        variance, entropy = self._calculate_metrics(image_np)
        
        print(f"Métricas para {image_path}:")
        print(f"  Varianza: {variance}")
        print(f"  Entropía: {entropy}")

        if variance < 40:  
            print(f"Advertencia: La imagen {image_path} podría estar oscurecida.")

    def save_photo(self, photo_number):
        """
        Captura y guarda una foto.
        """
        photo_path = self._get_photo_path(photo_number)
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(photo_path, frame)
            print(f"Foto {photo_path} guardada.")
            self._process_image(photo_path)
        else:
            print("Error capturando la foto.")
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
            print("Error al abrir la cámara.")
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
    