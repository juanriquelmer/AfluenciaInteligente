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
    def calculate_images_mean(inputs, from_path=True):
        """Calcula el promedio de los valores de los pixeles de todas las imagenes."""
        image_means = []
        for input_data in inputs:
            image = cv2.imread(input_data, cv2.IMREAD_GRAYSCALE) if from_path else input_data
            if image is not None:
                image_means.append(np.mean(image))
        overall_mean = np.mean(image_means) if len(image_means) > 0 else None
        return overall_mean

    @staticmethod
    def calculate_images_variance(inputs, from_path=True):
        """Calcula la varianza de los valores de los pixeles de todas las imagenes."""
        image_variances = []
        for input_data in inputs:
            image = cv2.imread(input_data, cv2.IMREAD_GRAYSCALE) if from_path else input_data
            if image is not None:
                image_variances.append(np.var(image))
        overall_variance = np.mean(image_variances) if len(image_variances) > 0 else None
        return overall_variance

    @staticmethod
    def calculate_average_ssim(image, inputs, from_path=True):
        """Calcula el SSIM promedio entre las imágenes."""
        ssim_values = []
        for input_data in inputs:
            compare_image = cv2.imread(input_data, cv2.IMREAD_GRAYSCALE) if from_path else input_data
            if compare_image is not None:
                ssim_value = ImageMetrics._calculate_ssim(image, compare_image)  # Make sure this method exists!
                ssim_values.append(ssim_value)
        average_ssim = np.mean(ssim_values) if ssim_values else None
        return average_ssim

    @staticmethod
    def check_sigma_criterion(last_image, inputs, from_path=True, multiplier=3):
        """Evalúa el criterio de las tres desviaciones estándar."""
        overall_mean = ImageMetrics.calculate_images_mean(inputs, from_path=from_path)
        overall_variance = ImageMetrics.calculate_images_variance(inputs, from_path=from_path)

        if overall_mean is None or overall_variance is None:
            return

        std_dev = np.sqrt(overall_variance)
        lower_bound = overall_mean - multiplier * std_dev
        upper_bound = overall_mean + multiplier * std_dev

        if last_image is not None:
            last_image_mean = np.mean(last_image)
            if last_image_mean < lower_bound or last_image_mean > upper_bound:
                return False  # Indicando que la imagen es una anomalía
        return True  # Indicando que la imagen está dentro del rango normal
