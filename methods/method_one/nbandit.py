import time
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import cv2
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
    count_cumulative,
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
    lr: float = 0.001,
) -> list:
    """Returns a training loss series, cumulative reward series, weight series, cumulative action sum series and episode duration series
    :param data: Pandas dataframe with a training sample
    :param weights: a list of action weights
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param lr: learning rate for the Keras classification model
    :return: a list with training loss series, cumulative reward series, weight series, cumulative action sum series and episode duration series
    """
    episodes = len(data)
    weights = weights2tensors(weights)
    action_count_stack = []
    running_cumulative_episode_actions_count = []
    running_reward = []
    running_cumulative_episode_reward = []
    loss_stack = []
    episode_duration = []

    for i in tqdm(data.itertuples(), total=episodes):

        start = time.time()

        image = cv2.imread(i.dataset_path, cv2.IMREAD_UNCHANGED)
        if image.shape != (100, 100):
            image = resize_image(image)
        action_idx, action, action_prob = generate_action(actions, weights, epsilon)
        action_count = tf.Variable(tf.zeros(len(actions)))
        action_count = tf.reshape(action_count[action_idx].assign(1), [1, 5])
        action_count_stack.append(action_count)
        running_cumulative_episode_actions_count.append(
            count_cumulative(
                action_count_stack, running_cumulative_episode_actions_count, True, True
            )
        )

        reward = env_interaction(image, i.tokens, action)
        running_reward.append(reward)
        running_cumulative_episode_reward.append(
            count_cumulative(running_reward, running_cumulative_episode_reward, True)
        )

        with tf.GradientTape() as tape:
            tape.watch(action_prob)
            loss = custom_loss(action_prob, reward)

        gradient = tape.gradient(tf.constant(loss), weights)
        optimizer = tf.keras.optimizers.Adagrad(lr)
        optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(gradient, weights) if grad is not None
        )

        loss_stack.append(loss)

        end = time.time()
        duration = end - start
        episode_duration.append(duration)

    return [
        loss_stack,
        running_reward,
        running_cumulative_episode_reward,
        episode_duration,
        running_cumulative_episode_actions_count,
    ]
