# Adding libraries required for text detection with Google Vision API
from google.cloud import vision
import io
import numpy as np
import cv2
from methods.utils.other import process_api_output


def baseline_text_detection(image: [str, np.ndarray], from_file: bool = True) -> list:
    """Detects text on image
    :param str image: path to a file with an image to be processed
    :returns str: string of words detected on the provided image
    """
    client = vision.ImageAnnotatorClient()

    if from_file:
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
