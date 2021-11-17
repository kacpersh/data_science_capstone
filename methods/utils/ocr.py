from google.cloud import vision
import io
import numpy as np
import cv2
from retrying import retry


def process_api_output(api_output: list) -> str:
    """Returns a list with processed Google Vision API output
    :param api_output: a list with raw Google Vision API output
    :return: a list with processed Google Vision API output
    """
    try:
        del api_output[0]
        return " ".join(api_output)
    except IndexError:
        return ""


@retry(stop_max_attempt_number=5, stop_max_delay=5000)
def baseline_text_detection(image: [str, np.ndarray]) -> str:
    """Detects text on image
    :param image: path to a file with an image to be processed or a numpy array with an image
    :returns: string of words detected on the provided image
    """
    client = vision.ImageAnnotatorClient()

    if type(image) is str:
        with io.open(image, "rb") as f:
            content = f.read()

        input = vision.Image(content=content)

    else:
        _, content = cv2.imencode(".png", image)
        input = vision.Image(content=content.tobytes())

    response = client.text_detection(image=input)
    texts = response.text_annotations

    return process_api_output([text.description for text in texts])
