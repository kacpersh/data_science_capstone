import tensorflow as tf
import numpy as np


def calculate_reward(
    baseline_ecr: float, rl_ecr: float, high_performance_reward: float = 100.0
) -> tf.Tensor:
    """Calculates a reward for the Reinforcement Learning model to feedback on the chosen action
    :param baseline_ecr: Character Error Rate calculated between the actual text on the image and predicted text on the
    non-processed image
    :param rl_ecr: Character Error Rate calculated between the actual text on the image and predicted text on the
    RL-processed image
    :param high_performance_reward: reward if Character Error Rate between the actual text on the image and predicted
    text on the RL-processed image is lower than between the former and predicted text on thenon-processed image
    :return: reward for the Reinforcement Learning model to feedback on the chosen action
    """
    if rl_ecr > baseline_ecr:
        return tf.cast(tf.Variable(-1 * rl_ecr), dtype=tf.float64)
    else:
        return tf.cast(tf.Variable(high_performance_reward), dtype=tf.float64)


def standardizer(input: [tf.Tensor, np.ndarray]) -> tf.Tensor:
    """Standardizes a Tensor or an array of values
    :param input: a flat Tensor or array with values to standardize
    :return: a Tensor with standardized values
    """
    eps = np.finfo(np.float32).eps.item()
    output = (input - tf.math.reduce_mean(input)) / (tf.math.reduce_std(input) + eps)
    return output


# Based on the tutorial from https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
def rewards2returns(
    rewards: [tf.Tensor, np.ndarray], gamma: float, standarization: bool = True
) -> tf.Tensor:
    """Converts a Tensor with rewards to a Tensor with expected returns
    :param rewards: a Tensor with rewards for conversion
    :param gamma: a decaying discount factor, the higher the value the more forward looking the less weight for future values
    :return: a Tensor with expected returns
    """
    rewards_len = len(rewards)
    rewards = [tf.reverse(x, [1]) for x in rewards]
    discounted_sum = 0.0
    expected_returns = []
    for i in tf.range(rewards_len):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        expected_returns.append(discounted_sum)
    expected_returns = [tf.reverse(x, [1]) for x in expected_returns]
    if standarization is True:
        expected_returns = standardizer(expected_returns)
    return expected_returns
