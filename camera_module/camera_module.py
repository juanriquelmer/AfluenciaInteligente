import os
import time
import cv2

from image_metrics import ImageMetrics

class CameraModule:
    """Clase para gestionar la cámara y procesar las imágenes capturadas."""
    def __init__(self, max_photos=100, camera_index=0, photo_directory="CameraModule_Log", capture_period=10):
        self.capturing = False  # Flag para saber si está capturando o no
        self.photo_count = 0  # Contador de fotos tomadas
        self.max_photos = max_photos  # Máximo de fotos que se pueden tomar
        self.photo_format = ".jpg"
        self.photo_directory = photo_directory
        self.camera_index = camera_index  # Índice de la cámara
        self.capture_period = capture_period  # Tiempo entre capturas

        # Crear directorio para las fotos si no existe
        if not os.path.exists(self.photo_directory):
            os.makedirs(self.photo_directory)

        self.image_metrics = ImageMetrics()

    # directory methods

    def _get_photo_path(self, photo_number):
        """Genera y retorna la ruta completa de la foto basada en el número proporcionado."""
        photo_name = f"{photo_number:03d}{self.photo_format}"
        return os.path.join(self.photo_directory, photo_name)

    def save_photo(self, photo_number):
        """Captura y guarda una foto."""
        photo_path = self._get_photo_path(photo_number)
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(photo_path, frame)
            print(f"Foto {photo_path} guardada.")
            self._evaluate_last_image_metrics()
        else:
            print("Error capturando la foto.")
            self._handle_error()

    def delete_photo(self, photo_number):
        """Elimina una foto basada en el número proporcionado."""
        photo_path = self._get_photo_path(photo_number)
        if os.path.exists(photo_path):
            os.remove(photo_path)
            print(f"Foto {photo_path} eliminada.")
        else:
            print("Error eliminando la foto.")
            self._handle_error()

    # capture methods

    def capture(self):
        """Comienza la captura de fotos en intervalos definidos por capture_period."""
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

    def _evaluate_last_image_metrics(self):
        """Evalúa las métricas de la última imagen capturada."""
        # Obtiene la ruta de la última imagen y la lee
        actual_image_path = self._get_photo_path(self.photo_count)
        actual_image = cv2.imread(actual_image_path, cv2.IMREAD_GRAYSCALE)

        # Calcula las métricas
        variance = ImageMetrics.calculate_variance(actual_image)
        entropy = ImageMetrics.calculate_entropy(actual_image)
        average_ssim = ImageMetrics.calculate_average_ssim(actual_image, self.photo_directory)
        noise_level = ImageMetrics.calculate_noise_level(actual_image)
        is_normal = ImageMetrics.check_sigma_criterion(actual_image, self.photo_directory, multiplier=1.5)

        print(f"  Varianza: {variance}")
        print(f"  Entropía: {entropy}")
        print(f"  SSIM medio: {average_ssim}")
        print(f"  Nivel de ruido: {noise_level}")
        
        if variance < 1000:
            print(f"Advertencia: La imagen {self.photo_count} podría estar oscurecida/obstruida.")
        if not is_normal:
            print(f"Advertencia: La imagen {self.photo_count} podría estar desviada de la normalidad.")
        
    def _handle_error(self):
        """Maneja los errores que pueden surgir durante la captura o gestión de fotos."""
        pass
