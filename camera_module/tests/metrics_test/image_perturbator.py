import cv2
import numpy as np

class ImagePerturbator:

    @staticmethod
    def brighten(image, gamma=1.5):
        """ Oscurece la imagen usando corrección gamma."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def darken(image, gamma=0.5):
        """ Aclara la imagen usando corrección gamma."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    @staticmethod
    def blur(image, kernel_size=(5,5)):
        """ Desenfoca la imagen """
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def reduce_contrast(image, alpha=0.5):
        """Reduce el contraste de la imagen."""
        # Calcula la media de luminancia de la imagen
        mean_luminance = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # Interpola la imagen entre sus valores actuales y la media de luminancia
        return cv2.convertScaleAbs((1 - alpha) * mean_luminance + alpha * image)