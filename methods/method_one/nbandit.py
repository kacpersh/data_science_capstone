# Adding libraries required for test image preprocessing
from random import randint
import pandas as pd
import numpy as np
from fastwer import score
from tqdm import tqdm
import tensorflow as tf
from methods.utils.ocr import baseline_text_detection
from methods.utils.rewards import calculate_reward
from methods.utils.actions import (
    scaleup_image,
    denoise_image,
    adaptive_thresholding,
    increase_brightness,
)


def baseline_cer(image_path: str, tokens: str) -> float:
    """Returns a Character Error Rate for text detected from an unprocessed image
    :param image_path: path to an image
    :param tokens: string with all words actually being on an image
    :return: Character Error Rate for text detected from an unprocessed image
    """
    extracted_tokens = baseline_text_detection(image_path)
    return score([extracted_tokens], [tokens], True)


def rl_cer(action, image_path: str, tokens: str) -> float:
    """Returns a Character Error Rate for text detected from a processed image
    :param action: function to process the image
    :param image_path: path to an image
    :param tokens: string with all words actually being on an image
    :return: Character Error Rate for text detected from a processed image
    """
    preprocessed_img = action(image_path)
    extracted_tokens = baseline_text_detection(preprocessed_img, False)
    return score([extracted_tokens], [tokens], True)


def generate_action(actions: list, weights: list, epsilon: float = 0.2) -> tuple:
    """Returns a function to process the image, ensures balancing out exploration and exploitation
    :param actions: a list of functions to process the image
    :param weights: a list of tensors with action weights
    :param epsilon: probability bar to select an action different from the optimal one
    :return: tuple with index of the action and function to process the image
    """
    if np.random.rand(1) < epsilon:
        action_idx = randint(0, 3)
        action = actions[action_idx]
    else:
        action_idx = weights.index(max(weights))
        action = actions[action_idx]
    return action_idx, action


def custom_loss(action_weight: float, reward: [int, float]) -> float:
    """Returns a loss given function/action weight and received reward
    :param action_weight: current action weight
    :param reward: reward received for using an action to preprocess an image
    :return: a loss value
    """
    return -(tf.math.log(action_weight) * reward)


def weights2tensors(weights: list) -> list:
    """Converts a list of floats to a list of tensors
    :param weights: list of action weights to convert
    :return: a list of tensor action weights
    """
    w1 = tf.Variable(weights[0])
    w2 = tf.Variable(weights[1])
    w3 = tf.Variable(weights[2])
    w4 = tf.Variable(weights[3])
    return [w1, w2, w3, w4]


def nbandit(
    data: pd.DataFrame,
    weights: list = [0.25, 0.25, 0.25, 0.25],
    epsilon: float = 0.2,
    actions: list = [
        scaleup_image,
        denoise_image,
        adaptive_thresholding,
        increase_brightness,
    ],
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(),
) -> list:
    """Returns a total reward and weight per action after the learning loop
    :param data: Pandas dataframe with a training sample
    :param weights: a list of action weights
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param optimizer: TensorFlow optimizer to be used in the processes of adjusting the weight
    :return: a list with training loss and weight and total reward  per action after the learning loop
    """
    episodes = len(data)
    weights_series = [weights]
    loss_series = [0]
    weights = weights2tensors(weights)

    for i in tqdm(data.itertuples(), total=episodes):
        action = generate_action(actions, weights, epsilon)

        baseline_score = baseline_cer(i.dataset_path, i.tokens)
        rl_score = rl_cer(action[1], i.dataset_path, i.tokens)

        reward = calculate_reward(baseline_score, rl_score)

        action_weight = weights[action[0]]

        with tf.GradientTape() as tape:
            tape.watch(action_weight)
            loss = custom_loss(action_weight, reward)

        gradient = tape.gradient(loss, weights)
        optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(gradient, weights) if grad is not None
        )

        loss_series.append(loss.numpy())
        weights_series.append([i.numpy() for i in weights])

    return [loss_series, weights_series]
