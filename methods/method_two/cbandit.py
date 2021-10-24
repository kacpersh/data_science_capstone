# Adding libraries required for test image preprocessing
import time
import pandas as pd
import numpy as np
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
    build_model,
    custom_loss,
    generate_action_contextual,
    resize_image,
    env_interaction,
    cumulative_action_count,
)


def cbandit(
    data: pd.DataFrame,
    model: tf.keras.Model = build_model(),
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
    """Returns a total reward and weight per action after the learning loop
    :param data: Pandas dataframe with a training sample
    :param model: a Keras classification model
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param optimizer: TensorFlow optimizer to be used in the processes of adjusting the weight
    :return: a list with training loss series, cumulative reward series, weight series, cumulative action sum series and episode duration series
    """
    episodes = len(data)
    loss_series = [0]
    reward_series = [0]
    action_cumulative_count = [[0, 0, 0, 0, 0]]
    duration_series = [0]

    for i in tqdm(data.itertuples(), total=episodes):

        start = time.time()

        image = resize_image(i.dataset_path)
        state = np.expand_dims(image, 0)
        pred_actions = model(state)
        action_idx, action = generate_action_contextual(pred_actions, actions, epsilon)
        action_cumulative_sum = cumulative_action_count(
            action_cumulative_count, action_idx
        )
        action_cumulative_count.append(action_cumulative_sum)
        reward = env_interaction(image, i.tokens, action)
        reward_series.append(reward)
        eval_action = tf.slice(tf.reshape(pred_actions, -1), [action_idx], [1])

        with tf.GradientTape() as tape:
            tape.watch(eval_action)
            loss = custom_loss(eval_action, reward)

        gradient = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradient, model.trainable_weights)
            if grad is not None
        )

        loss_series.append(float(loss.numpy()))

        end = time.time()
        episode_duration = end - start
        duration_series.append(episode_duration)

    return [loss_series, reward_series, action_cumulative_count, duration_series]
