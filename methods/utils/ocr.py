# Adding libraries required for text detection with Google Vision API
from google.cloud import vision
import io
import numpy as np
import cv2


def process_api_output(api_output: str) -> str:
    """Returns a list with processed Google Vision API output
    :param api_output: a list with raw Google Vision API output
    :return: a list with processed Google Vision API output
    """
    try:
        del api_output[0]
        return " ".join(api_output)
    except IndexError:
        return ""


def baseline_text_detection(image: [str, np.ndarray]) -> list:
    """Detects text on image
    :param str image: path to a file with an image to be processed or a numpy array with an image
    :param bool from_file: boolean input to indicate if an image needs to be read from path or is already contained in an NumPy array
    :returns str: string of words detected on the provided image
    """
    client = vision.ImageAnnotatorClient()

    if type(image) is str:
        # Opening file
        with io.open(image, "rb") as f:
            content = f.read()

        # Loading file to Google Vision API
        input = vision.Image(content=content)

    else:
        # Loading np.ndarray to Google Vision API
        _, content = cv2.imencode(".png", image)
        input = vision.Image(content=content.tobytes())

    # Generating predictions
    response = client.text_detection(image=input)
    texts = response.text_annotations

    return process_api_output([text.description for text in texts])
