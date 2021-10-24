def calculate_reward(
    baseline_ecr: float, rl_ecr: float, high_performance_reward: int = 100
) -> float:
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
        return -1 * rl_ecr
    else:
        return high_performance_reward


def cumulative_rewards(data: list) -> list:
    """Calculates a cumulative reward series
    :param data: a list of rewards collected over a number of episodes
    :return: a list with a cumulative reward series
    """
    cumulative_reward_series = [data[0]]
    [
        cumulative_reward_series.append(cumulative_reward_series[-1] + i)
        for i in data[1:]
    ]
    return cumulative_reward_series
