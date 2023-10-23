import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2
import numpy as np

class ImageEnhancer:
    """
    A class that provides methods to enhance images.
    """

    @staticmethod
    def enhance_contrast(image_path):
        """
        Enhances the contrast of an image.

        Parameters:
        image_path (str): The path to the image file.

        Returns:
        numpy.ndarray: The enhanced image.
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([40, 40, 40], dtype=np.uint8)
        mask_black = cv2.inRange(img_rgb, lower_black, upper_black)

        alpha = 2.0 
        img_enhanced = img_rgb.copy()
        img_enhanced[mask_black == 255] = np.clip(alpha * img_enhanced[mask_black == 255], 0, 255)

        return img_enhanced

    @staticmethod
    def apply_clahe(image_path):
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

        Parameters:
        image_path (str): The path to the image file.

        Returns:
        numpy.ndarray: The enhanced image.
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        cl = clahe.apply(l_channel)
        clahe_img = cv2.merge((cl, a_channel, b_channel))
        result = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB)

        return result

    @staticmethod
    def adjust_gamma(image_path, gamma=1.0):
        """
        Adjusts the gamma of an image.

        Parameters:
        image_path (str): The path to the image file.
        gamma (float): The gamma value. Default is 1.0.

        Returns:
        numpy.ndarray: The enhanced image.
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(img_rgb, table)

        return result

    @staticmethod
    def apply_gaussian_blur(image_path, ksize=(5, 5)):
        """
        Applies Gaussian blur to an image.

        Parameters:
        image_path (str): The path to the image file.
        ksize (tuple): The kernel size. Default is (5, 5).

        Returns:
        numpy.ndarray: The enhanced image.
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = cv2.GaussianBlur(img_rgb, ksize, 0)
        return result

    @staticmethod
    def apply_sharpening(image_path):
        """
        Applies sharpening to an image.

        Parameters:
        image_path (str): The path to the image file.

        Returns:
        numpy.ndarray: The enhanced image.
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        kernel = np.array([
            [ 0, -0.25,  0],
            [-0.25,  2, -0.25],
            [ 0, -0.25,  0]
        ])
        result = cv2.filter2D(img_rgb, -1, kernel)
        return result

