import hashlib
import pickle
from functools import wraps
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from random import randint
from fastwer import score
from collections import Counter
from methods.utils.ocr import baseline_text_detection
from methods.utils.rewards import calculate_reward


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


def save_pickle(item, path: str):
    """Saves a pickled object
    :param obj item: an object to be saved
    :param str path: a string with a path location
    """
    with open(path, "wb") as filehandle:
        pickle.dump(item, filehandle)


def load_pickle(path: str):
    """Opens pickled files
    :param str path: path to the pickled file
    :return: a pickled file
    """
    with open(path, "rb") as filehandle:
        return pickle.load(filehandle)


def weights2tensors(weights: list) -> list:
    """Converts a list of floats to a list of tensors
    :param weights: list of action weights to convert
    :return: a list of tensor action weights
    """
    w1 = tf.Variable(weights[0], dtype=tf.float64)
    w2 = tf.Variable(weights[1], dtype=tf.float64)
    w3 = tf.Variable(weights[2], dtype=tf.float64)
    w4 = tf.Variable(weights[3], dtype=tf.float64)
    w5 = tf.Variable(weights[4], dtype=tf.float64)
    return [w1, w2, w3, w4, w5]


def resize_image(image: [str, np.ndarray], dim: tuple = (100, 100)) -> np.ndarray:
    """Resizes image to a specified shape
    :param image: path to a file with an image to be processed or a numpy array with an image
    :param dim: a tuple with a new image shape
    :returns: a numpy array with a resized image
    """
    if type(image) is str:
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def generate_action(actions: list, weights: list, epsilon: float = 0.2) -> tuple:
    """Returns a function to process the image, ensures balancing out exploration and exploitation
    :param actions: a list of functions to process the image
    :param weights: a list of tensors with action weights
    :param epsilon: probability bar to select an action different from the optimal one
    :return: tuple with index of the action and function to process the image
    """
    if np.random.rand(1) < epsilon:
        action_idx = randint(0, len(weights) - 1)
        action_prob = weights[action_idx]
        action = actions[action_idx]
    else:
        action_idx = weights.index(max(weights))
        action_prob = weights[action_idx]
        action = actions[action_idx]
    return action_idx, action, action_prob


def baseline_cer(image: [str, np.ndarray], tokens: str) -> float:
    """Returns a Character Error Rate for text detected from an unprocessed image
    :param image: path to a file with an image to be processed or a numpy array with an image
    :param tokens: string with all words actually being on an image
    :return: Character Error Rate for text detected from an unprocessed image
    """
    extracted_tokens = baseline_text_detection(image)
    return score([extracted_tokens], [tokens], True)


def rl_cer(
    action, image: [str, np.ndarray], tokens: str, return_state: bool = False
) -> [tuple, float]:
    """Returns a Character Error Rate for text detected from a processed image
    :param action: function to process the image
    :param image: path to a file with an image to be processed or a numpy array with an image
    :param tokens: string with all words actually being on an image
    :param return_state: option to return the preprocessed image
    :return: Character Error Rate for text detected from a processed image
    """
    preprocessed_img = action(image)
    extracted_tokens = baseline_text_detection(preprocessed_img)
    rl_score = score([extracted_tokens], [tokens], True)
    if return_state is True:
        return rl_score, preprocessed_img
    else:
        return rl_score


def custom_loss(action_weight: tf.Tensor, reward: float) -> tf.Tensor:
    """Returns a loss given function/action weight and received reward
    :param action_weight: current action weight
    :param reward: reward received for using an action to preprocess an image
    :return: a loss value
    """
    return -(tf.math.log(action_weight) * reward)


def env_interaction(
    image: np.ndarray, tokens: str, action, return_state: bool = False
) -> [tf.Tensor, np.ndarray]:
    """Simulates interaction of the action with the environment, an image, and produces a reward
    :param image: path to a file with an image to be processed or a numpy array with an image
    :param tokens: string with all words actually being on an image
    :param action: function to process the image
    :param return_state: boolean if the function should also return a preprocessed image
    :return: a reward value and optionally a NumPy array with the preprocessed image
    """
    baseline_score = baseline_cer(image, tokens)
    if return_state is True:
        rl_score, preprocessed_img = rl_cer(action, image, tokens, return_state)
        reward = calculate_reward(baseline_score, rl_score)
        return reward, preprocessed_img
    else:
        rl_score = rl_cer(action, image, tokens, return_state)
        reward = calculate_reward(baseline_score, rl_score)
        return reward


def build_model(
    shape: tuple = (100, 100, 3),
    kernel: tuple = (3, 3),
    pool_size: tuple = (2, 2),
    no_classes: int = 5,
) -> tf.keras.Model:
    """Generates a TensorFlow CNN model later used to identify a state in the contextual bandit and policy-based agent
    :param shape: an input shape of a state, in this case an image
    :param kernel: a kernel size to be used across all the Convolution layers
    :param pool_size: a pool size to be used across all the Pooling layers
    :param no_classes: a number of classes, in this case actions, the model will need to predict for
    :return: a TensorFlow CNN model
    """
    inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Conv2D(100, kernel, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size)(x)
    x = tf.keras.layers.Conv2D(200, kernel, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size)(x)
    x = tf.keras.layers.Conv2D(200, kernel, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    outputs = tf.keras.layers.Dense(no_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def generate_action_contextual(
    actions_logits: tf.Tensor,
    actions: list,
    epsilon: float,
) -> tuple:
    """Returns a function to process the image, ensures balancing out exploration and exploitation
    :param actions_logits: an output from the DNN model with action logit probabilities
    :param actions: a list of functions to process the image
    :param epsilon: probability bar to select an action different from the optimal one
    :return: tuple with index of the action and function to process the image
    """
    if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < epsilon:
        action_idx = tf.random.categorical(actions_logits, 1)[0, 0]
        action = actions[int(action_idx)]
    else:
        actions_probs = tf.nn.softmax(actions_logits)
        action_idx = tf.argmax(actions_probs, -1)[0]
        action = actions[int(action_idx)]
    return action_idx, action


def count_cumulative(
    running_total: list,
    running_cumulative: list,
    action_count: bool = False,
    cbandit: bool = False,
) -> [tf.Tensor, list]:
    """Counts cumulative values of the running input, either reward or action count
    :param running_total: a list with a series Tensor values to calculate the cumulative output
    :param running_cumulative: a list where the cumulative values should be aggregated
    :param action_count: an indicator whether the function is used to count the cumulative values of action count or not
    :return: current cumulative value to append to running_cumulative
    """
    try:
        last_cumulative = running_cumulative[-1]
        new_total = running_total[-1]
        if action_count is True:
            return tf.squeeze(tf.math.add(last_cumulative, new_total))
        else:
            return [tf.squeeze(tf.math.add(last_cumulative, new_total))]
    except IndexError:
        if cbandit is True:
            return tf.squeeze(running_total[-1])
        else:
            return running_total[-1]


def count_action_combinations(array: list, actions: list) -> dict:
    """Counts unique action combinations and outputs them in a sorted dictionary
    :param array: a list with arrays or Tensors with action combinations to process
    :param actions: a list of functions to process the image
    :return: a sorted dictionary with the unique action combinations from training
    """
    action_names = [i.__name__ for i in actions]
    array_processed = [np.where(i.numpy() >= 1, 1, 0) for i in array]
    array_processed = [list(np.argwhere(i == 1).flatten()) for i in array_processed]
    array_processed = [
        [action_names[action] for action in combination]
        for combination in array_processed
    ]
    array_processed = ["\n".join(combination) for combination in array_processed]
    action_combination_count = dict(Counter(array_processed))
    action_combination_count_sorted = dict(
        sorted(action_combination_count.items(), key=lambda item: item[1])
    )
    return action_combination_count_sorted
