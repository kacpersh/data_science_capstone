# Adding libraries required for test image preprocessing
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance


# TECHNIQUE 1: Scale up image
def scaleup_image(
    image_path: str, xscale: [int, float] = 2, yscale: [int, float] = 2
) -> np.ndarray:
    """Scales image up, 2x by default
    :param str image_path: path to a file with an image to be processed
    :param int/float xscale: scale magnitude over X-axis
    :param int/float yscale: scale magnitude over Y-axis
    :returns: a processed scaled-up image
    """
    image = cv2.imread(image_path)
    scaled_image = cv2.resize(
        image, None, fx=xscale, fy=yscale, interpolation=cv2.INTER_LINEAR
    )
    return scaled_image


# TECHNIQUE 2: De-noising image
def denoise_image(image_path: str, filter_strength: [int, float] = 10) -> np.ndarray:
    """Removes noise form a provided image
    :param str image_path: path to a file with an image to be processed
    :param int/float filter_strength: denoising filter strength
    :returns: a processed image with lowered noise
    """
    image = cv2.imread(image_path)
    denoised_image = cv2.fastNlMeansDenoisingColored(image, h=filter_strength)
    return denoised_image


# TECHNIQUE 3: Binary thresholding
def adaptive_thresholding(
    image_path: str, max_value: int = 255, block_size: int = 11, constant: int = 2
) -> np.ndarray:
    """Performs thresholding using the adaptive thresholding method
    :param str image_path: path to a file with an image to be processed
    :param str max_value: max value assigned to a pixel if its value is above threshold
    :param str block_size: size of pixel neighbourhood used to calculate threshold
    :param str constant: constant to be subtracted from the weighted mean
    :returns: a processed threshold image
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(gray_image, 3)
    threshold_image = cv2.adaptiveThreshold(
        blurred_image,
        max_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )
    return threshold_image


# TECHNIQUE 4: Brightness increase
def increase_brightness(
    image_path: str, brightness_increase: [int, float] = 1.5
) -> np.ndarray:
    """Increases brigthness of an image
    :param str image_path: path to a file with an image to be processed
    :param int/float brightness_increase: brightness increase magnitude
    :returns: a processed threshold image
    """
    image = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(brightness_increase)
    brightened_image = np.array(brightened_image)
    return brightened_image
