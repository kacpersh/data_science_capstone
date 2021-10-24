from PIL import Image, ImageCms

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


def get_text_area_summary(bounding_box_array: np.ndarray, num_of_words: int) -> tuple:
    """Calculates the summary statistics of text area in an image.
    :param np.ndarray bounding_box_array: An array of bounding box co-ordinates, one per word.
    :param int num_of_words: An integer for the total number of words in an image.
    :returns: A tuple containing the total text area, largest text area, smallest text area and
        average text area per image.
    """
    bounding_box_array = bounding_box_array.T
    areas = []

    # For images with only one word, bounding box array has one less dimension.
    # We need to rearrange the co-ordinates in the anti-clockwise direction for shoelace algorithm to work.
    # For each word, calculate the bounding box area.
    for i in range(num_of_words):
        if num_of_words == 1:
            x, y = (
                bounding_box_array[[0, 3, 2, 1], 0],
                bounding_box_array[[0, 3, 2, 1], 1],
            )
        else:
            x, y = (
                bounding_box_array[i][[0, 3, 2, 1], 0],
                bounding_box_array[i][[0, 3, 2, 1], 1],
            )
        area = get_bounding_box_area(x, y)
        areas.append(area)
    return sum(areas), max(areas), min(areas), np.mean(areas)


def get_mean_image_brightness(img_path: str) -> float:
    """Calculates average brightness of an image. We convert the image to LAB space, extract the luminance channel and
    then calculate the average luminance  value.
    :param str img_path: path to a file with an image to be processed.
    :returns: A floating value ranging from 0 to 255 representing the average luminance value of the iamge
    """
    with Image.open(img_path) as im:
        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        rgb2lab = ImageCms.buildTransformFromOpenProfiles(
            srgb_profile, lab_profile, "RGB", "LAB"
        )
        lab_representation = ImageCms.applyTransform(im, rgb2lab)
        l, a, b = lab_representation.split()
        mean_brightness = np.mean(np.array(l.getdata()))
        return mean_brightness


def classify_by_luminance(
    image_luminance: float, lower_bound: float = 60, upper_bound: float = 150
) -> str:
    """Classifies if an image is of low, medium or high luminance
    :param float image_luminance: Luminance value of an image.
    :param float lower_bound: Lower bound for luminance. Values below lower bound will be classified as 'Low'.
    :param float upper_bound: Higher bound for luminance. Values above higher bound will be classified as 'Large'.
    :returns: A string representing the category to which a particular luminance value belongs.
    """
    if image_luminance <= lower_bound:
        return "Low"
    elif lower_bound < image_luminance <= upper_bound:
        return "Medium"
    elif image_luminance > upper_bound:
        return "Large"


def classify_by_resolution(
    image_resolution: int, lower_bound: int = 200000, upper_bound: int = 270000
) -> str:
    """Classifies if an image is of low, medium or high luminance
    :param float image_resolution: Resolution an image. It is the product of image height and width.
    :param float lower_bound: Lower bound for resolution. Values below lower bound will be classified as 'Low'.
    :param float upper_bound: Higher bound for resolution. Values above higher bound will be classified as 'Large'.
    :returns: A string representing the category to which a particular image resolution value belongs.
    """
    if image_resolution <= lower_bound:
        return "Low"
    elif lower_bound < image_resolution <= upper_bound:
        return "Medium"
    elif image_resolution > upper_bound:
        return "Large"


def classify_by_fontsize(
    image_total_font_area: int, lower_bound: int = 5, upper_bound: int = 20
) -> str:
    """Classifies if an image is of low, medium or high luminance
    :param float image_total_font_area: Total font area (sum of bounding box areas) per image.
    :param float lower_bound: Lower bound for font area. Values below lower bound will be classified as 'Low'.
    :param float upper_bound: Higher bound for font area. Values above higher bound will be classified as 'Large'.
    :returns: A string representing the category to which a particular image font area value belongs.
    """
    if image_total_font_area <= lower_bound:
        return "Low"
    elif lower_bound < image_total_font_area <= upper_bound:
        return "Medium"
    elif image_total_font_area > upper_bound:
        return "Large"
