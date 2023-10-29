import os
import time
import cv2
from image_metrics import ImageMetrics


class CameraModule:
    """Clase para gestionar la cámara y procesar las imágenes capturadas."""
    def __init__(self, max_photos=100, camera_index=0, photo_directory="CameraModule_Log", capture_period=10, num_images_to_analyze=100):
        self.capturing = False  # Flag para saber si está capturando o no
        self.photo_count = 0  # Contador de fotos tomadas
        self.max_photos = max_photos  # Máximo de fotos que se pueden tomar
        self.photo_format = ".jpg"
        self.photo_directory = photo_directory
        self.camera_index = camera_index  # Índice de la cámara
        self.capture_period = capture_period  # Tiempo entre capturas
        self.num_images_to_analyze = num_images_to_analyze  # Número de imágenes a analizar

        # Crear directorio para las fotos si no existe
        if not os.path.exists(self.photo_directory):
            os.makedirs(self.photo_directory)

    # directory methods

    def _get_photo_path(self, photo_number):
        """Genera y retorna la ruta completa de la foto basada en el número proporcionado."""
        photo_name = f"{photo_number:03d}{self.photo_format}"
        return os.path.join(self.photo_directory, photo_name)
    
    def _get_latest_image_paths(self):
        """Obtiene las rutas de las últimas imágenes basadas en self.num_images_to_analyze y teniendo en cuenta el ciclo de numeración."""
        image_numbers = [(self.photo_count - i - 1) % self.max_photos for i in range(self.num_images_to_analyze)]
        image_paths = [self._get_photo_path(n) for n in image_numbers if os.path.exists(self._get_photo_path(n))]
        
        return image_paths

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
        """Evalúa las métricas de la última imagen capturada y las compara con las de las imágenes anteriores."""
        
        actual_image_path = self._get_photo_path(self.photo_count)
        print(f"  Imagen actual: {actual_image_path}")
        actual_image_metrics = ImageMetrics.get_metrics(actual_image_path)
        print(f"  Métricas de la última imagen:")
        print(f"Brillo: {actual_image_metrics['brightness']}" + f"Varianza del Laplaciano: {actual_image_metrics['variance_of_laplacian']}" + f"Entropía del histograma: {actual_image_metrics['histogram_entropy']}" + f"Contraste de la imagen: {actual_image_metrics['image_contrast']}")

        last_images_paths = self._get_latest_image_paths()
        print(f"  Últimas imágenes: {last_images_paths}")
        
        metrics_to_evaluate = [ImageMetrics.brightness_metric, ImageMetrics.variance_of_laplacian, ImageMetrics.histogram_entropy,  ImageMetrics.image_contrast]
        metrics_mapping = {
            ImageMetrics.brightness_metric: 'brightness',
            ImageMetrics.variance_of_laplacian: 'variance_of_laplacian',
            ImageMetrics.histogram_entropy: 'histogram_entropy',
            ImageMetrics.image_contrast: 'image_contrast'
        }
        results = {}
        for metric_func in metrics_to_evaluate:
            metric_name = metrics_mapping.get(metric_func, "Unknown Metric")
            results[metric_name] = ImageMetrics.analyze_image_set(last_images_paths, metric_func=metric_func)
        
        print(f"  Resultados:")
        for metric_name, metric_results in results.items():
            print(f"  {metric_name}:")
            print(f"    Promedio: {metric_results['mean']}")
            print(f"    Percentil10: {metric_results['percentile10']}")
            print(f"    Percentil90: {metric_results['percentile90']}")

            if metric_results['percentile10'] > actual_image_metrics[metric_name] or metric_results['percentile90'] < actual_image_metrics[metric_name]:
                print(f"    ALERTA: La imagen actual está fuera del rango normal para {metric_name}.")

    def _handle_error(self):
        """Maneja los errores que pueden surgir durante la captura o gestión de fotos."""
        pass
