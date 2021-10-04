# Adding libraries required for test image preprocessing
import hashlib
import pickle
from functools import wraps
import pandas as pd


def as_bytes(func, targets=str):
    """Converts function's string inputs to bytes
    :return: bytes representation of function's string inputs
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = [arg.encode() if isinstance(arg, targets) else arg for arg in args]
        new_kwargs = {
            key: value.encode() if isinstance(value, targets) else value
            for key, value in kwargs.items()
        }
        return func(*new_args, **new_kwargs)

    return wrapper


@as_bytes
def hash_path(some_val: [str, bytes]) -> bytes:
    """Converts strings to hash digest
    :param some_val: thing to hash, can be str or bytes
    :return: sha256 hash digest of some_val, type bytes
    """
    return hashlib.sha256(some_val).digest()


def get_img_id(image_path: str) -> str:
    """Returns image id in form of sha256 hash digest hexadecimal representation for image path
    :param image_path: username that will be hashed, can be str or bytes
    :return: image id in form of sha256 hash digest, type bytes
    """
    return hash_path(image_path.lower()).hex()


def filtering_folders(dataframe: pd.DataFrame, folder_name: int) -> pd.DataFrame:
    """Returns a dataset with a random sample of observations filtered by background/folder
    :param dataframe: Pandas DataFrame to filter and sample
    :param folder_name: background/folder type
    :return: a dataset with a random sample of filtered observations
    """
    df_folder_filtered = dataframe[dataframe.folder == folder_name]
    return df_folder_filtered


def filtering_focus_types(dataframe: pd.DataFrame, focus_type: str) -> pd.DataFrame:
    """Returns a dataset with a random sample of observations filtered by an area of interest
    :param dataframe: Pandas DataFrame to filter and sample
    :param focus_type: area of interest to be kept at 'Low', will keep the remaining two areas at 'Medium'
    :return: a dataset with a random sample of filtered observations
    """
    df_folder_filtered = dataframe[dataframe[focus_type] == "Low"]
    columns = list(df_folder_filtered.columns)
    remove = ["dataset_path", "folder", "tokens"] + [focus_type]
    remaining_types = [c for c in columns if c not in remove]
    df_folder_filtered = df_folder_filtered[
        (df_folder_filtered[remaining_types[0]] == "Medium")
        & (df_folder_filtered[remaining_types[1]] == "Medium")
    ]
    return df_folder_filtered


class Sampling:
    """Returns a dataset with a random sample of observations filtered by an area of interest
    :param dataframe_path: path to a Pandas DataFrame to sample
    :param sample_size: number of observations to be included in the random sample
    :return: a dataset with a random sample observations, they could be filtered depending on the chosen method
    """

    def __init__(self, dataframe_path, sample_size):
        self.df = pd.read_parquet(dataframe_path)
        self.sample_size = sample_size

    def filter_a(self, folder_name: int) -> pd.DataFrame:
        """Returns a dataset with a random sample of observations filtered by background/folder
        :param folder_name: background/folder type
        :return: a dataset with a random sample of filtered observations
        """
        df_folder_filtered = filtering_folders(self.df, folder_name)
        df_sample = df_folder_filtered.sample(self.sample_size)
        return df_sample

    def filter_b(self, focus_type: str) -> pd.DataFrame:
        """Returns a dataset with a random sample of observations filtered by an area of interest
        :param focus_type: area of interest to be kept at 'Low', will keep the remaining two areas at 'Medium'
        :return: a dataset with a random sample of filtered observations
        """
        df_folder_filtered = filtering_focus_types(self.df, focus_type)
        df_sample = df_folder_filtered.sample(self.sample_size)
        return df_sample

    def filter_ab(self, folder_name: int, focus_type: str) -> pd.DataFrame:
        """Returns a dataset with a random sample of observations filtered by folder name and area of interest
        :param folder_name: background/folder type
        :param focus_type: area of interest to be kept at 'Low', will keep the remaining two areas at 'Medium'
        :return: a dataset with a random sample of filtered observations
        """
        df_folder_filtered = filtering_folders(self.df, folder_name)
        df_folder_filtered = filtering_focus_types(df_folder_filtered, focus_type)
        df_sample = df_folder_filtered.sample(self.sample_size)
        return df_sample

    def no_filter(self) -> pd.DataFrame:
        """Returns a dataset with a random sample of observations with no filtering applied
        :return: a dataset with a random sample from all observations
        """
        df_sample = self.df.sample(self.sample_size)
        return df_sample


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


def save_pickle(item, path):
    """Saves a pickled object
    :param obj item: an object to be saved
    :param str path: a string with a path location
    """
    with open(path, "wb") as filehandle:
        pickle.dump(item, filehandle)


def load_pickle(path):
    """Opens pickled files
    :param str path: path to the pickled file
    :return: a pickled file
    """
    with open(path, "rb") as filehandle:
        return pickle.load(filehandle)
