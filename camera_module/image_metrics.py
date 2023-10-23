import cv2
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim


class ImageMetrics:
    def __init__(self):
        pass

    @staticmethod
    def calculate_variance(image):
        return np.var(image)

    @staticmethod
    def calculate_entropy(image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        return stats.entropy(hist)

    @staticmethod
    def calculate_ssim(image_a, image_b):
        return ssim(image_a, image_b)

    @staticmethod
    def calculate_noise_level(image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift))
        noise_level = np.max(magnitude_spectrum) - np.mean(magnitude_spectrum)
        return noise_level

    @staticmethod
    def calculate_images_mean(image_paths):
        """Calcula el promedio de los valores de los pixeles de todas las imagenes en image_paths"""
        image_means = []
        for photo_path in image_paths:
            image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_means.append(np.mean(image))
        overall_mean = np.mean(image_means) if len(image_means) > 0 else None
        return overall_mean

    @staticmethod
    def calculate_images_variance(image_paths):
        """Calcula la varianza de los valores de los pixeles de todas las imagenes en image_paths"""
        image_variances = []
        for photo_path in image_paths:
            image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_variances.append(np.var(image))
        overall_variance = np.mean(image_variances) if len(image_variances) > 0 else None
        return overall_variance

    @staticmethod
    def calculate_average_ssim(image, image_paths):
        """Calcula el SSIM promedio entre las imágenes."""
        ssim_values = []

        for photo_path in image_paths:
            image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)

            if image is not None and image is not None:
                ssim_value = ImageMetrics._calculate_ssim(image, image)
                ssim_values.append(ssim_value)

        average_ssim = np.mean(ssim_values) if ssim_values else None
        return average_ssim

    @staticmethod
    def check_sigma_criterion(last_image, image_paths, multiplier=3):
        """Evalúa el criterio de las tres desviaciones estándar."""
        overall_mean = ImageMetrics.calculate_images_mean(image_paths)
        overall_variance = ImageMetrics.calculate_images_variance(image_paths)

        # Si no se tienen datos suficientes, no evaluar el criterio
        if overall_mean is None or overall_variance is None:
            return

        std_dev = np.sqrt(overall_variance)
        lower_bound = overall_mean - multiplier * std_dev
        upper_bound = overall_mean + multiplier * std_dev

        # Calcula la media de la última imagen
        if last_image is not None:
            last_image_mean = np.mean(last_image)

            # Evaluar el criterio de las desviaciones estándar
            if last_image_mean < lower_bound or last_image_mean > upper_bound:
                return False  # Indicando que la imagen es una anomalía

        return True  # Indicando que la imagen está dentro del rango normal
