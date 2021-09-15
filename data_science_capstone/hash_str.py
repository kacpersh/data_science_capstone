import os
from hashlib import sha256
from typing import Union


def get_csci_salt() -> bytes:
    """Returns the appropriate salt for CSCI E-29

    :return: bytes representation of the CSCI salt
    """
    return bytes.fromhex(os.environ["CSCI_SALT"])


def asbytes(input: Union[str, bytes]) -> bytes:
    """Converts function's string inputs to bytes

    :return: bytes representation of function's string inputs
    """
    return input.encode() if isinstance(input, str) else input


def hash_str(some_val: Union[str, bytes], salt: Union[str, bytes] = "")\
        -> bytes:
    """Converts strings to hash digest

    :param some_val: thing to hash, can be str or bytes
    :param salt: Add randomness to the hashing, can be str or bytes
    :return: sha256 hash digest of some_val with salt, type bytes
    """
    return sha256(asbytes(salt) + asbytes(some_val)).digest()


def get_user_id(username: str) -> str:
    """Returns first 8 characters of sha256 hash digest hexadecimal representation
    for username with CSCI salt

    :param username: username that will be hashed, can be str or bytes
    :return: first 8 characters of sha256 hash digest, type bytes
    """
    return hash_str(username.lower(), salt=get_csci_salt()).hex()[:8]
