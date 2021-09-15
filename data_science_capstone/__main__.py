from data_science_capstone.hash_str import get_csci_salt, get_user_id, hash_str


def get_user_hash(username, salt=None):
    """Returns first 8 characters of sha256 hash digest hexadecimal
    representation for username

    :param username: username that will be hashed, can be str or bytes
    :param salt: Add randomness to the hashing, can be str or bytes
    :return: first 8 characters of sha256 hash digest, type bytes
    """
    salt = salt or get_csci_salt()
    return hash_str(username, salt=salt)


# Removed the below block of code from coverage.
# Applied functions come from other modules and were tested already.
if __name__ == "__main__":  # pragma: no cover

    for user in ["gorlins", "kacpersh"]:
        print("Id for {}: {}".format(user, get_user_id(user)))
