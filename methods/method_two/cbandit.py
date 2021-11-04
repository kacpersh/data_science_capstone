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
    build_model,
    generate_action_contextual,
    resize_image,
    env_interaction,
    count_cumulative,
)
from methods.utils.rewards import standardizer


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
    lr: float = 0.001,
) -> list:
    """Returns a total reward and weight per action after the learning loop
    :param data: Pandas dataframe with a training sample
    :param model: a Keras classification model
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param lr: learning rate for the Keras classification model
    :return: a list with training loss series, cumulative reward series, weight series, cumulative action sum series and episode duration series
    """
    episodes = len(data)
    action_count_stack = []
    running_cumulative_episode_actions_count = []
    running_reward = []
    running_cumulative_episode_reward = []
    loss_stack = []
    episode_duration = []

    actions_logits_update_batch = []
    encoded_actions_update_batch = []
    rewards_update_batch = []

    batch_size = 0
    for i in tqdm(data.itertuples(), total=episodes):

        start = time.time()

        image = cv2.imread(i.dataset_path, cv2.IMREAD_UNCHANGED)
        if image.shape != (100, 100):
            image = resize_image(image)
        state = tf.expand_dims(image, 0)

        with tf.GradientTape() as tape:
            actions_logits = standardizer(model(state))
            actions_logits_update_batch.append(tf.squeeze(actions_logits))
            action_idx, action = generate_action_contextual(
                actions_logits, actions, epsilon
            )

            action_count = tf.Variable(
                tf.zeros(actions_logits.shape[1], dtype=tf.float64)
            )
            action_count = action_count[action_idx].assign(1)
            encoded_actions_update_batch.append(action_count)
            action_count_stack.append(action_count)

            running_cumulative_episode_actions_count.append(
                count_cumulative(
                    action_count_stack,
                    running_cumulative_episode_actions_count,
                    True,
                    True,
                )
            )

            reward = env_interaction(image, i.tokens, action)
            rewards_update_batch.append(reward)
            running_reward.append(reward)
            running_cumulative_episode_reward.append(
                count_cumulative(
                    running_reward, running_cumulative_episode_reward, True
                )
            )

            actions_logits_update_batch_stack = tf.cast(
                tf.stack(actions_logits_update_batch), dtype=tf.float64
            )
            encoded_actions_update_batch_stack = tf.cast(
                tf.stack(encoded_actions_update_batch), dtype=tf.float64
            )
            rewards_update_batch_stack = tf.expand_dims(
                standardizer(tf.cast(tf.stack(rewards_update_batch), dtype=tf.float64)),
                1,
            )

            loss = -tf.reduce_sum(
                tf.multiply(
                    tf.nn.softmax_cross_entropy_with_logits(
                        encoded_actions_update_batch_stack,
                        logits=actions_logits_update_batch_stack,
                    ),
                    rewards_update_batch_stack,
                )
            )
            loss_stack.append(loss)

            if batch_size % 5 == 0 or batch_size == (episodes - 1):

                gradient = tape.gradient(loss, model.trainable_weights)
                optimizer = tf.keras.optimizers.Adagrad(lr)
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))

                actions_logits_update_batch.clear()
                encoded_actions_update_batch.clear()
                rewards_update_batch.clear()

        end = time.time()
        duration = end - start
        episode_duration.append(duration)
        batch_size += 1

    return [
        loss_stack,
        running_reward,
        running_cumulative_episode_reward,
        episode_duration,
        running_cumulative_episode_actions_count,
    ]
