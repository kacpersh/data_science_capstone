"""Module file with helper functions for generating attributes relevant for EDA/data preperation."""
from nltk.tokenize import word_tokenize


def clean_text_labels(list_of_words: list) -> list:
    """Removes white spaces, new line, carriage return and other word seperators from a list of text labels.
    :param list list_of_words: A list with all text labels on a word.
    :returns: A list of words after removing the white spaces.
    """
    sentence = " ".join(list_of_words)
    return word_tokenize(sentence)


def find_character_type(character: str) -> str:
    """Classifies a given character to Non alpha numeric, upper case letter, lower case letter or digit.
    :param str character: A character which has to be categorized.
    :returns: A string with the appropriate category.
    """
    if not character.isalnum():
        return "Non Alpha Numeric"
    elif character.isupper():
        return "Upper Case Letter"
    elif character.islower():
        return "Lower Case Letter"
    elif character.isdigit():
        return "Digit"
    else:
        return character
