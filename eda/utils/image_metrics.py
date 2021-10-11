from PIL import Image

import numpy as np


def get_image_entropy_and_size(img_path: str) -> tuple:
    """Calculates the complexity and size of any image.
    :param str img_path: path to a file with an image to be processed
    :returns: A tuple with image entropy scope, image height and image width.
    """
    with Image.open(img_path) as im:
        entropy = im.entropy()
        height = im.height
        width = im.width
    return entropy, height, width


def get_bounding_box_area(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the area of a quadrilateral using shoelace algorithm from x and y co-ordinates.
    :param np.ndarray x: An array of ints representing the x co-ordinates of the quadrilateral
    :param np.ndarray y: An array of ints representing the y co-ordinates of the quadrilateral
    :returns: A floating point number which represents the area of the quadrilateral.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
