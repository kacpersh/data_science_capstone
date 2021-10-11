from PIL import Image


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
