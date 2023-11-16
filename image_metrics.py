import cv2
import numpy as np
from scipy import stats

class ImageMetrics:

    @staticmethod
    def load_image(input_data, file_path=True):
        """Carga una imagen dada una ruta o directamente."""
        if file_path:
            return cv2.imread(input_data)
        return input_data
    
    def load_images(input_data, file_path=True):
        """Carga una lista de imágenes dada una ruta o directamente."""
        if file_path:
            return [cv2.imread(image) for image in input_data]
        return input_data

    @staticmethod
    def brightness_metric(input_data, file_path=True):
        """Devuelve el brillo promedio de la imagen."""
        image = ImageMetrics.load_image(input_data, file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    @staticmethod
    def variance_of_laplacian(input_data, file_path=True):
        """Devuelve la varianza del Laplaciano (medida de enfoque)."""
        image = ImageMetrics.load_image(input_data, file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def histogram_entropy(input_data, file_path=True):
        """Devuelve la entropía del histograma de la imagen."""
        image = ImageMetrics.load_image(input_data, file_path)
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        hist /= hist.sum()
        entropy = -np.sum(hist*np.log2(hist + np.finfo(float).eps))
        return entropy

    @staticmethod
    def image_contrast(input_data, file_path=True):
        """Devuelve el contraste de la imagen."""
        image = ImageMetrics.load_image(input_data, file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)  

    @staticmethod
    def get_metrics(input_data, file_path=True):
        """Devuelve las métricas de una imagen."""
        image = ImageMetrics.load_image(input_data, file_path)
        metrics = {
            'brightness': ImageMetrics.brightness_metric(input_data, file_path),
            'variance_of_laplacian': ImageMetrics.variance_of_laplacian(input_data, file_path),
            'histogram_entropy': ImageMetrics.histogram_entropy(input_data, file_path),
            'image_contrast': ImageMetrics.image_contrast(input_data, file_path)
        }
        return metrics
    
    @staticmethod
    def analyze_image_set(images, file_path=True, metric_func=None):
        """Analiza un conjunto de imágenes y decide si es necesario tomar otra foto o emitir una alerta."""
        if not metric_func:
            raise ValueError("Por favor proporciona una función de métrica.")

        metric_values = [metric_func(img, file_path) for img in images]

        mean_value = np.mean(metric_values)
        median_value = np.median(metric_values)
        std_value = np.std(metric_values)
        
        percentile10 = np.percentile(metric_values, 10)
        percentile25 = np.percentile(metric_values, 25)
        percentile75 = np.percentile(metric_values, 75)
        percentile90 = np.percentile(metric_values, 90)

        mean_minus_3std = mean_value - 3 * std_value
        mean_plus_3std = mean_value + 3 * std_value

        return {
            "mean": mean_value,
            "median": median_value,
            "std": std_value,
            "percentile10": percentile10,
            "percentile25": percentile25,
            "percentile75": percentile75,
            "percentile90": percentile90,
            "mean_minus_3std": mean_minus_3std,
            "mean_plus_3std": mean_plus_3std,
            "all_values": metric_values
        }
