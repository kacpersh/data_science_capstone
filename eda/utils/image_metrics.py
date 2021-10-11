from PIL import Image

import numpy as np
from nltk.tokenize import word_tokenize


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


def clean_text_labels(list_of_words: list) -> list:
    """Removes white spaces, new line, carriage return and other word seperators from a list of text labels.
    :param list list_of_words: A list with all text labels on a word.
    :returns: A list of words after removing the white spaces.
    """
    sentence = " ".join(list_of_words)
    return word_tokenize(sentence)
