# Adding libraries required for test image preprocessing
import time
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from methods.utils.actions import (
    scaleup_image,
    denoise_image,
    adaptive_thresholding,
    increase_brightness,
    no_action,
)
from methods.utils.other import (
    generate_action,
    custom_loss,
    weights2tensors,
    resize_image,
    env_interaction,
    cumulative_action_count,
)


def nbandit(
    data: pd.DataFrame,
    weights: list = [0.2, 0.2, 0.2, 0.2, 0.2],
    epsilon: float = 0.2,
    actions: list = [
        scaleup_image,
        denoise_image,
        adaptive_thresholding,
        increase_brightness,
        no_action,
    ],
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(),
) -> list:
    """Returns a training loss series, cumulative reward series, weight series, cumulative action sum series and episode duration series
    :param data: Pandas dataframe with a training sample
    :param weights: a list of action weights
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param optimizer: TensorFlow optimizer to be used in the processes of adjusting the weight
    :return: a list with training loss series, cumulative reward series, weight series, cumulative action sum series and episode duration series
    """
    episodes = len(data)
    loss_series = [0]
    reward_series = [0]
    weights = weights2tensors(weights)
    action_cumulative_count = [[0, 0, 0, 0, 0]]
    duration_series = [0]

    for i in tqdm(data.itertuples(), total=episodes):

        start = time.time()

        image = resize_image(i.dataset_path)
        action_idx, action = generate_action(actions, weights, epsilon)
        action_cumulative_sum = cumulative_action_count(
            action_cumulative_count, action_idx
        )
        action_cumulative_count.append(action_cumulative_sum)
        reward = env_interaction(image, i.tokens, action)
        reward_series.append(reward)
        eval_action = weights[action_idx]

        with tf.GradientTape() as tape:
            tape.watch(eval_action)
            loss = custom_loss(eval_action, reward)

        gradient = tape.gradient(loss, weights)
        optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(gradient, weights) if grad is not None
        )

        loss_series.append(float(loss.numpy()))

        end = time.time()
        episode_duration = end - start
        duration_series.append(episode_duration)

    return [loss_series, reward_series, action_cumulative_count, duration_series]
