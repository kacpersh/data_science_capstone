# The Actor-Critic model in this module is a modified version from TensorFlow guide at:
# https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
import time
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import cv2
from methods.utils.other import (
    resize_image,
    env_interaction,
    count_cumulative,
    count_action_combinations,
    build_model,
    generate_action_ac,
    ac_loss,
)
from methods.utils.actions import (
    scaleup_image,
    denoise_image,
    adaptive_thresholding,
    increase_brightness,
    no_action,
)
from methods.utils.rewards import rewards2returns


class ActorCritic(tf.keras.Model):
    """Custom Actor-Critic Keras model, based on the classification model used in the previous methods, with two outputs, Actor and Critic"""

    def __init__(self, no_classes: int = 5):
        super(ActorCritic, self).__init__()

        self.shared_layers = [layer for layer in build_model(actor_critic=True).layers]
        self.actor = tf.keras.layers.Dense(no_classes, name="actor")
        self.critic = tf.keras.layers.Dense(1, name="critic")

    def call(self, input):
        x = input
        for layer in self.shared_layers:
            x = layer(x)
        return self.actor(x), self.critic(x)


def run_step(
    data: pd.DataFrame,
    model: tf.keras.Model,
    epsilon: float,
    actions: list,
    max_steps: int,
) -> tuple:
    """Manages the process of interaction with the environment: state processing, action selection and reward generation
    :param data: Pandas dataframe with a training sample
    :param model: a custom Actor-Critic Keras model
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param max_steps: a maximum number of preprocessing steps the model can take on one image
    :return: a tuple with action probability values, encoded action values, reward, step count and episode action count
    """
    actions_probs_stack = []
    values_stack = []
    rewards_stack = []
    encoded_actions_stack = []
    step_count = 0

    image = cv2.imread(data.dataset_path, cv2.IMREAD_UNCHANGED)

    for _ in tf.range(max_steps):

        if image.shape != (100, 100):
            image = resize_image(image)
        state = tf.cast(tf.expand_dims(image, 0), dtype=tf.float64)

        actions_logits, value = model(state)
        actions_probs, action_idx, action = generate_action_ac(
            actions_logits, actions, epsilon
        )
        actions_probs_stack.append(actions_probs)

        action_count = tf.Variable(tf.zeros(actions_logits.shape[1], dtype=tf.float64))
        action_count = action_count[action_idx].assign(1)
        encoded_actions_stack.append(action_count)

        values_stack.append(tf.squeeze(value))

        reward, preprocessed_img = env_interaction(image, data.tokens, action, True)
        rewards_stack.append(reward)

        step_count += 1

        if action.__name__ == "adaptive_thresholding":
            preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.cv2.COLOR_GRAY2RGB)
        image = preprocessed_img

        if action.__name__ == "no_action":
            break

    actions_probs_stack = tf.cast(tf.stack(actions_probs_stack), dtype=tf.float64)
    values_stack = tf.cast(tf.stack(values_stack), dtype=tf.float64)
    rewards_stack = tf.cast(tf.stack(rewards_stack), dtype=tf.float64)
    encoded_actions_stack = tf.cast(tf.stack(encoded_actions_stack), dtype=tf.float64)
    episode_actions_count = tf.math.reduce_sum(encoded_actions_stack, axis=0)

    return (
        actions_probs_stack,
        values_stack,
        rewards_stack,
        encoded_actions_stack,
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
    """Manages the training loops and updates to the custom Actor-Critic Keras model
    :param data: Pandas dataframe with a training sample
    :param model: a custom Actor-Critic Keras model
    :param epsilon: probability bar to select an action different from the optimal one
    :param actions: a list of functions to process the image
    :param max_steps: a maximum number of preprocessing steps the model can take on one image
    :param gamma: a decaying discount factor, the higher the value the more forward looking the less weight for future values
    :param lr: learning rate for the custom Actor-Critic Keras model
    :return: a tuple with loss, step count, total episode reward, episode duration and episode actions count
    """
    start = time.time()

    total_episode_reward = []
    episode_duration = []

    with tf.GradientTape() as tape:
        (
            actions_probs_stack,
            values_stack,
            rewards_stack,
            encoded_actions_stack,
            step_count,
            episode_actions_count,
        ) = run_step(data, model, epsilon, actions, max_steps)

        total_episode_reward.append(tf.math.reduce_sum(rewards_stack))
        returns_stack = rewards2returns(rewards_stack, gamma, False)

        loss = ac_loss(actions_probs_stack, values_stack, returns_stack)

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


def actor_critic(
    data: pd.DataFrame,
    model: tf.keras.Model = ActorCritic(),
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
    :param model: a custom Actor-Critic Keras model
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
