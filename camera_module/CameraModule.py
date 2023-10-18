import os
import time

import cv2
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim


class CameraModule:
    """
    Clase para gestionar la cámara y su funcionalidad de captura.
    """
    def __init__(self, camera_index=0):
        self.capturing = False  # Flag para saber si está capturando o no
        self.photo_count = 0  # Contador de fotos tomadas
        self.max_photos = 20  # Máximo de fotos que se pueden tomar
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

    def _calculate_ssim(self, image_a, image_b):
        # Calcula el SSIM entre dos imágenes
        return ssim(image_a, image_b)

    def _evaluate_ssim_last_image(self):
        # Compara la última imagen con todas las imágenes anteriores y obtiene una media de los valores SSIM
        last_image_path = self._get_photo_path(self.photo_count - 1)
        last_image = cv2.imread(last_image_path, cv2.IMREAD_GRAYSCALE)

        if last_image is None:
            return

        ssim_values = []
        for photo_number in range(self.photo_count - 1):  # -1 para excluir la última imagen
            photo_path = self._get_photo_path(photo_number)
            image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                ssim_value = self._calculate_ssim(last_image, image)
                ssim_values.append(ssim_value)

        average_ssim = np.mean(ssim_values)
        print(f"SSIM medio de la última imagen con respecto a las anteriores: {average_ssim}")

    def _calculate_noise_level(self, image):
        # Utiliza FFT para identificar el ruido en una imagen
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift))

        # Un ruido fuerte se manifestará como un pico en el espectro de frecuencias
        noise_level = np.max(magnitude_spectrum) - np.mean(magnitude_spectrum)
        return noise_level

    def _evaluate_noise_last_image(self):
        # Calcula el nivel de ruido de la última imagen
        last_image_path = self._get_photo_path(self.photo_count - 1)
        last_image = cv2.imread(last_image_path, cv2.IMREAD_GRAYSCALE)

        if last_image is None:
            return

        noise_level = self._calculate_noise_level(last_image)
        print(f"Nivel de ruido de la última imagen: {noise_level}")

    def _process_image(self, image_path):
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        variance, entropy = self._calculate_metrics(image_np)

        print(f"Métricas para {image_path}:")
        print(f"  Varianza: {variance}")
        print(f"  Entropía: {entropy}")

        if variance < 1000:
            print(f"Advertencia: La imagen {image_path} podría estar oscurecida/obstruida.")

        # Adicional: Evaluar el criterio de las tres desviaciones estándar
        self._check_three_sigma_criterion()
        self._evaluate_ssim_last_image()
        self._evaluate_noise_last_image()

    def _calculate_image_stats(self):
        # Calcular media y varianza para cada imagen
        image_means = []
        image_variances = []
        for photo_number in range(self.photo_count):
            photo_path = self._get_photo_path(photo_number)
            image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_means.append(np.mean(image))
                image_variances.append(np.var(image))

        # Calcular la media y la varianza de las medias/varianzas de las imágenes
        overall_mean = np.mean(image_means) if len(image_means) > 0 else None
        overall_variance = np.mean(image_variances) if len(image_variances) > 0 else None
        return overall_mean, overall_variance

    def _check_three_sigma_criterion(self):
        # Calcular media y varianza de las medias/varianzas de todas las imágenes
        overall_mean, overall_variance = self._calculate_image_stats()

        # Si no se tienen datos suficientes, no evaluar el criterio
        if overall_mean is None or overall_variance is None:
            return

        std_dev = np.sqrt(overall_variance)
        lower_bound = overall_mean - 1.5 * std_dev
        upper_bound = overall_mean + 1.5 * std_dev

        # Calcula la media de la última imagen
        last_image_path = self._get_photo_path(self.photo_count - 1)
        last_image = cv2.imread(last_image_path, cv2.IMREAD_GRAYSCALE)
        if last_image is not None:
            last_image_mean = np.mean(last_image)

            # Evaluar el criterio de las 3 desviaciones estándar
            if last_image_mean < lower_bound or last_image_mean > upper_bound:
                print("Advertencia: La última imagen capturada se desvía significativamente del comportamiento habitual.")

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

if __name__ == "__main__":
    camera = CameraModule()
    camera.capture()