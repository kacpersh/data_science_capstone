import time
from tqdm import tqdm
import pandas as pd
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
    resize_image,
    env_interaction,
    generate_action_contextual,
    count_cumulative,
    count_action_combinations,
    build_model,
)
from methods.utils.rewards import rewards2returns, standardizer


def run_step(
    data: pd.DataFrame,
    model: tf.keras.Model,
    epsilon: float,
    actions: list,
    max_steps: int,
) -> tuple:
    """Manages the process of interaction with the environment: state processing, action selection and reward generation
    :param data: Pandas dataframe with a training sample
    :param model: a Keras classification model
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param max_steps: a maximum number of preprocessing steps the model can take on one image
    :return: a tuple with action logit values, encoded action values, reward, step count and episode action count
    """
    actions_logits_stack = []
    rewards_stack = []
    encoded_actions_stack = []
    step_count = 0
    image = cv2.imread(data.dataset_path, cv2.IMREAD_UNCHANGED)

    for _ in tf.range(max_steps):

        if image.shape != (100, 100):
            image = resize_image(image)
        state = tf.expand_dims(image, 0)
        # Standardization is required to control the loss and gradient
        actions_logits = standardizer(model(state))
        actions_logits_stack.append(tf.squeeze(actions_logits))
        action_idx, action = generate_action_contextual(
            actions_logits, actions, epsilon
        )

        action_count = tf.Variable(tf.zeros(actions_logits.shape[1], dtype=tf.float64))
        action_count = action_count[action_idx].assign(1)
        encoded_actions_stack.append(action_count)

        reward, preprocessed_img = env_interaction(image, data.tokens, action, True)
        rewards_stack.append(reward)

        step_count += 1

        if action.__name__ == "adaptive_thresholding":
            preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.cv2.COLOR_GRAY2RGB)
        image = preprocessed_img

        if action.__name__ == "no_action":
            break

    actions_logits_stack = tf.cast(tf.stack(actions_logits_stack), dtype=tf.float64)
    encoded_actions_stack = tf.cast(tf.stack(encoded_actions_stack), dtype=tf.float64)
    episode_actions_count = tf.math.reduce_sum(encoded_actions_stack, axis=0)
    rewards_stack = tf.cast(tf.stack(rewards_stack), dtype=tf.float64)

    return (
        actions_logits_stack,
        encoded_actions_stack,
        rewards_stack,
        step_count,
        episode_actions_count,
    )


def run_episode(
    data: pd.DataFrame,
    model: tf.keras.Model,
    epsilon: float,
    actions: list,
    max_steps: int,
    gamma: float,
    lr: float,
) -> tuple:
    """Manages the training loops and updates to the Keras classification model
    :param data: Pandas dataframe with a training sample
    :param model: a Keras classification model
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param max_steps: a maximum number of preprocessing steps the model can take on one image
    :param gamma: a decaying discount factor, the higher the value the more forward looking the less weight for future values
    :param lr: learning rate for the Keras classification model
    :return: a tuple with loss, step count, total episode reward, episode duration and episode actions count
    """
    start = time.time()

    total_episode_reward = []
    episode_duration = []

    with tf.GradientTape() as tape:
        (
            actions_logits_stack,
            encoded_actions_stack,
            rewards_stack,
            step_count,
            episode_actions_count,
        ) = run_step(data, model, epsilon, actions, max_steps)
        total_episode_reward.append(tf.math.reduce_sum(rewards_stack))
        returns_stack = tf.expand_dims(rewards2returns(rewards_stack, gamma), 1)
        # Different loss comapred to the one of the nbandit method to prevent inf and/or nan loss outputs
        loss = -tf.reduce_sum(
            tf.multiply(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=encoded_actions_stack, logits=actions_logits_stack
                ),
                returns_stack,
            )
        )

    gradient = tape.gradient(loss, model.trainable_weights)
    optimizer = tf.keras.optimizers.Adagrad(lr)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    end = time.time()
    duration = end - start
    episode_duration.append(duration)

    return (
        loss,
        step_count,
        total_episode_reward,
        episode_duration,
        episode_actions_count,
    )


def pb_agent(
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
    max_steps: int = 5,
    gamma: float = 0.9,
    lr: float = 0.001,
) -> list:
    """Manages the entire training process across all steps and episodes
    :param data: Pandas dataframe with a training sample
    :param model: a Keras classification model
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param max_steps: a maximum number of preprocessing steps the model can take on one image
    :param gamma: a decaying discount factor, the higher the value the more forward looking the less weight for future values
    :param lr: learning rate for the Keras classification model
    :return: a list with loss, running step count, running total episode reward, running cumulative total episode reward,
    running episode duration, running cumulative_episode_actions_count and actions combinations count
    """
    episodes = len(data)
    loss_series = []
    running_step_count = []
    running_total_episode_reward = []
    running_cumulative_episode_reward = []
    running_episode_duration = []
    running_episode_actions_count = []
    running_cumulative_episode_actions_count = []

    for i in tqdm(data.itertuples(), total=episodes):
        (
            loss,
            step_count,
            total_episode_reward,
            episode_duration,
            episode_actions_count,
        ) = run_episode(i, model, epsilon, actions, max_steps, gamma, lr)
        loss_series.append(loss)
        running_step_count.append(step_count)
        running_total_episode_reward.append(total_episode_reward)
        running_cumulative_episode_reward.append(
            count_cumulative(
                running_total_episode_reward, running_cumulative_episode_reward
            )
        )
        running_episode_duration.append(episode_duration)
        running_episode_actions_count.append(episode_actions_count)
        running_cumulative_episode_actions_count.append(
            count_cumulative(
                running_episode_actions_count,
                running_cumulative_episode_actions_count,
                True,
            )
        )

    actions_combinations_count = count_action_combinations(
        running_episode_actions_count, actions
    )

    return [
        loss_series,
        running_step_count,
        running_total_episode_reward,
        running_cumulative_episode_reward,
        running_episode_duration,
        running_cumulative_episode_actions_count,
        actions_combinations_count,
    ]
