import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

@staticmethod
def enhance_edges_with_hog(image_input):
    """
    Visualize edges using Histogram of Oriented Gradients (HOG).

    Parameters:
    - image_input (str or numpy.ndarray): The path to the image file or a numpy array.

    Returns:
    numpy.ndarray: The visualization of HOG descriptor.
    """
    img_gray = cv2.cvtColor(ImageEnhancer._load_image(image_input), cv2.COLOR_RGB2GRAY)
    
    # Compute HOG descriptor and visualize it
    hog_image = hog(img_gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=False)[1]
    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_image_rescaled


"""
A class that provides methods to enhance images.

Methods:
- apply_better_black_regions(image_input, alpha=2.0)
- apply_clahe(image_input)
- adjust_gamma(image_input, gamma=1.0)
- apply_gaussian_blur(image_input, ksize=(5, 5))
- apply_sharpening(image_input)
- split_image(image_input)
- apply_bilateral_filter(image_input, d=15, sigma_color=75, sigma_space=75)
- apply_unsharp_mask(image_input, ksize=(5, 5), alpha=1.5, beta=-0.5)
"""
class ImageEnhancer:
    """
    A class that provides methods to enhance images.
    """

    @staticmethod
    def _load_image(image_input):
        """
        Load image from either a path or a numpy array.

        Parameters:
        image_input (str or numpy.ndarray): The path to the image file or a numpy array.

        Returns:
        numpy.ndarray: The loaded image in RGB format.
        """
        if isinstance(image_input, str):  # If it's a path, read the image
            img = cv2.imread(image_input, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):  # If it's already a numpy array
            # Assume that numpy arrays passed are always in RGB format
            img_rgb = image_input
        else:
            raise ValueError("Invalid input type. Provide either a path or a numpy ndarray.")
        
        return img_rgb



    @staticmethod
    def apply_better_black_regions(image_input, alpha=2.0):
        img_rgb = ImageEnhancer._load_image(image_input)

        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([40, 40, 40], dtype=np.uint8)
        mask_black = cv2.inRange(img_rgb, lower_black, upper_black)

        alpha = alpha
        img_enhanced = img_rgb.copy()
        img_enhanced[mask_black == 255] = np.clip(alpha * img_enhanced[mask_black == 255], 0, 255)

        result_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

        return img_enhanced

    @staticmethod
    def apply_clahe(image_input):
        img_rgb = ImageEnhancer._load_image(image_input)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        cl = clahe.apply(l_channel)
        clahe_img = cv2.merge((cl, a_channel, b_channel))
        result = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    @staticmethod
    def adjust_gamma(image_input, gamma=1.0):
        img_rgb = ImageEnhancer._load_image(image_input)

        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(img_rgb, table)

        return result

    @staticmethod
    def apply_gaussian_blur(image_input, ksize=(5, 5)):
        img_rgb = ImageEnhancer._load_image(image_input)

        result = cv2.GaussianBlur(img_rgb, ksize, 0)

        return result

    @staticmethod
    def apply_sharpening(image_input):
        img_rgb = ImageEnhancer._load_image(image_input)

        kernel = np.array([
            [ 0, -0.25,  0],
            [-0.25,  2, -0.25],
            [ 0, -0.25,  0]
        ])
        result = cv2.filter2D(img_rgb, -1, kernel)
        return result
    
    @staticmethod
    def split_image(image_input):
        """
        Split the image into 4 parts from the center.

        Parameters:
        - image_input (str or numpy.ndarray): The path to the image file or a numpy array.

        Returns:
        - List[numpy.ndarray]: List containing the 4 split images.
        """
        img_rgb = ImageEnhancer._load_image(image_input)

        h, w, _ = img_rgb.shape

        top_left = img_rgb[0:h//2, 0:w//2]
        top_right = img_rgb[0:h//2, w//2:w]
        bottom_left = img_rgb[h//2:h, 0:w//2]
        bottom_right = img_rgb[h//2:h, w//2:w]

        return [top_left, top_right, bottom_left, bottom_right]
    
    @staticmethod
    def apply_bilateral_filter(image_input, d=15, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter to the image.

        Parameters:
        - image_input (str or numpy.ndarray): The path to the image file or a numpy array.
        - d (int): Diameter of each pixel neighborhood that is used during filtering.
        - sigma_color (float): Filter sigma in color space. Larger value means that farther colors within the pixel neighborhood will be mixed together.
        - sigma_space (float): Filter sigma in coordinate space. Larger value means that farther pixels will influence each other.

        Returns:
        numpy.ndarray: The filtered image.
        """
        img_rgb = ImageEnhancer._load_image(image_input)
        return cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)

    @staticmethod
    def apply_unsharp_mask(image_input, ksize=(5, 5), alpha=1.5, beta=-0.5):
        """
        Apply unsharp mask to the image.

        Parameters:
        - image_input (str or numpy.ndarray): The path to the image file or a numpy array.
        - ksize (tuple): Size of the Gaussian kernel used for blurring.
        - alpha (float): Weight for the original image.
        - beta (float): Weight for the blurred image to be subtracted.

        Returns:
        numpy.ndarray: The sharpened image.
        """
        img_rgb = ImageEnhancer._load_image(image_input)
        blurred = cv2.GaussianBlur(img_rgb, ksize, 0)
        sharpened = cv2.addWeighted(img_rgb, alpha, blurred, beta, 0)
        return sharpened

    @staticmethod
    def apply_canny(image_input, low_threshold=50, high_threshold=150):
        """
        Apply Canny edge detection to the image.

        Parameters:
        - image_input (str or numpy.ndarray): The path to the image file or a numpy array.
        - low_threshold (int): Lower bound for the edge intensity gradient.
        - high_threshold (int): Upper bound for the edge intensity gradient.

        Returns:
        numpy.ndarray: The edge-detected image.
        """
        img_gray = cv2.cvtColor(ImageEnhancer._load_image(image_input), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, low_threshold, high_threshold)
        return edges


        
    @staticmethod
    def enhance_edges_with_hog(image_input):
        """
        Visualize edges using Histogram of Oriented Gradients (HOG).

        Parameters:
        - image_input (str or numpy.ndarray): The path to the image file or a numpy array.

        Returns:
        numpy.ndarray: The visualization of HOG descriptor.
        """
        img_gray = cv2.cvtColor(ImageEnhancer._load_image(image_input), cv2.COLOR_RGB2GRAY)
        
        # Compute HOG descriptor and visualize it
        hog_image = hog(img_gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)[1]
        
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        return hog_image_rescaled

