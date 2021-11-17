import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
from matplotlib import gridspec
import tensorflow as tf
from methods.utils.other import moving_average


def plot_time_series(
    data: list,
    graph_path: str,
    ylabel: str,
    title: str,
    xlabel: str = "No. of episodes",
):
    """Displays and saves a graph showing changes of selected time series data during the training process
    :param data: list with loss after each episode
    :param graph_path: path to save the graph
    :param ylabel: Y-axis label
    :param xlabel: X-axis label
    :param title: graph title
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(data)), data, linewidth=3)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()
    plt.savefig(graph_path)


def plot_cumulative_action_count(
    data: list,
    graph_path: str,
    actions: list = [
        "scaleup",
        "denoise",
        "thresholding",
        "brightness",
        "no_action",
    ],
    xlabel: str = "No. of episodes",
    ylabel: str = "Cumulative action count",
    title: str = "Changes of cumulative action count over no. of episodes",
):
    """Displays and saves a plot of cumulative action counts during the training process
    :param data: a list of lists with action weights for each episode
    :param graph_path: path to save the graph
    :param actions: a list of functions used as actions in the training process
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param title: graph title
    """
    c1 = [i[0] for i in data]
    c2 = [i[1] for i in data]
    c3 = [i[2] for i in data]
    c4 = [i[3] for i in data]
    c5 = [i[4] for i in data]
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(c1)), c1, linewidth=3)
    plt.plot(range(0, len(c2)), c2, linewidth=3)
    plt.plot(range(0, len(c3)), c3, linewidth=3)
    plt.plot(range(0, len(c4)), c4, linewidth=3)
    plt.plot(range(0, len(c5)), c5, linewidth=3)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(actions)
    plt.show()
    plt.savefig(graph_path)


def plot_cumulative_combination_count(
    data: dict,
    graph_path: str,
    ylabel: str = "Count of unique action combinations",
    title: str = "Count of top 5 unique action combinations from the training session",
):
    """Displays and saves a plot of the top 5 action combinations during the training process
    :param data: an ordered dictionary of action combinations and respective counts
    :param graph_path: path to save the graph
    :param ylabel: Y-axis label
    :param title: graph title
    """
    data = list(reversed(list(data.items())))[0:5]
    plt.figure(figsize=(20, 10))
    plt.bar([i[0] for i in data], [i[1] for i in data], linewidth=3)
    if len(data) < 5:
        plt.title(
            f"Count of top {len(data)} unique action combinations from the training session",
            fontsize=14,
            fontweight="bold",
        )
    else:
        plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12)
    plt.show()
    plt.savefig(graph_path)


def prepare_cumulative_action_count(
    data: [list, np.ndarray, tf.Tensor],
    title: str,
    axes: plt.axes,
    xlabel: bool,
    ylabel: bool,
    **kwargs,
) -> plt.plot:
    """Prepares a subplot with cumulative action count plot to later be included to an encompassing plot
    :param data: a series of data to be displayed on a histogram
    :param title: graph title
    :param axes: a subplot location
    :param xlabel: a boolean value if X-axis label should be included
    :param ylabel: a boolean value if Y-axis label should be included
    :return: a subplot
    """
    plt = axes
    c1 = [i[0] for i in data]
    c2 = [i[1] for i in data]
    c3 = [i[2] for i in data]
    c4 = [i[3] for i in data]
    c5 = [i[4] for i in data]
    plt.plot(range(0, len(c1)), c1, linewidth=3)
    plt.plot(range(0, len(c2)), c2, linewidth=3)
    plt.plot(range(0, len(c3)), c3, linewidth=3)
    plt.plot(range(0, len(c4)), c4, linewidth=3)
    plt.plot(range(0, len(c5)), c5, linewidth=3)
    plt.set_title(title, fontweight="bold")
    if xlabel is True:
        plt.set_xlabel(kwargs.get("xlabel_input"), fontsize=12)
    if ylabel is True:
        plt.set_ylabel(kwargs.get("ylabel_input"), fontsize=12)
    actions = ["scaleup", "denoise", "thresholding", "brightness", "no_action"]
    plt.legend(actions)
    return plt


def prepare_cumulative_combination_count(
    data: [list, np.ndarray, tf.Tensor],
    title: str,
    axes: plt.axes,
    ylabel: bool,
    **kwargs,
) -> plt.plot:
    """Prepares a subplot with cumulative action combination count plot to later be included to an encompassing plot
    :param data: a series of data to be displayed on a histogram
    :param title: graph title
    :param axes: a subplot location
    :param ylabel: a boolean value if Y-axis label should be included
    :return: a bar subplot
    """
    plt = axes
    data = list(reversed(list(data.items())))[0:5]
    plt.bar([i[0] for i in data], [i[1] for i in data], linewidth=3)
    plt.set_title(title, fontweight="bold")
    plt.xaxis.set_major_locator(FixedLocator([i for i in range(len(data))]))
    plt.set_xticklabels([i[0] for i in data], rotation=20, fontsize=6)
    if ylabel is True:
        plt.set_ylabel(kwargs.get("ylabel_input"), fontsize=10)
    return plt


def plot_summary_series(
    data: [list, np.ndarray, tf.Tensor],
    title: str,
    ylabel: str,
    graph_path: str,
    xlabel: str = "No. of episodes",
):
    """Displays and saves a graph showing changes of selected time series data during the training process
    :param data: list with loss after each episode
    :param title: graph title
    :param ylabel: Y-axis label
    :param xlabel: X-axis label
    :param graph_path: path to save the graph
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(data[0])), data[0], linewidth=3, label="nbandit")
    plt.plot(range(0, len(data[1])), data[1], linewidth=3, label="contextual bandit")
    plt.plot(
        range(0, len(data[2])),
        data[2],
        linewidth=3,
        label="policy-based (episode mean)",
    )
    plt.plot(
        range(0, len(data[3])),
        data[3],
        linewidth=3,
        label="actor-critic (episode mean)",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.show()
    plt.savefig(graph_path)


def plot_subplots(
    data: [list, np.ndarray, tf.Tensor],
    quadruple: bool,
    title: str,
    function,
    ylabel_input: str,
    graph_path: str,
    **kwargs,
):
    """Displays and saves a graph with two or four subplots
    :param data: list with loss after each episode
    :param quadruple: a boolean to indicate if there should be four subplots included
    :param title: graph title
    :param function: a Python function which should prepare the subplots
    :param ylabel_input: Y-axis label for the subplots
    :param graph_path: path to save the graph
    """
    if quadruple is True:
        fig = plt.figure(figsize=(10, 10))
        p1 = function(
            data[0],
            "Nbandit",
            fig.add_subplot(221),
            False,
            True,
            ylabel_input=ylabel_input,
        )
        p3 = function(data[1], "Contextual bandit", fig.add_subplot(222), False, False)
        p2 = function(
            data[2],
            "Policy-based",
            fig.add_subplot(223),
            True,
            True,
            ylabel_input=ylabel_input,
            xlabel_input=kwargs.get("xlabel_input"),
        )
        p4 = function(
            data[3],
            "Actor-critic",
            fig.add_subplot(224),
            True,
            False,
            xlabel_input=kwargs.get("xlabel_input"),
        )
    else:
        fig = plt.figure(figsize=(10, 5))
        if kwargs.get("combinations") is True:
            p1 = function(
                data[0],
                "Policy-based",
                fig.add_subplot(121),
                True,
                ylabel_input=ylabel_input,
            )
            p2 = function(data[1], "Actor-critic", fig.add_subplot(122), False)
        else:
            p1 = function(
                data[0],
                "Policy-based",
                fig.add_subplot(121),
                False,
                True,
                ylabel_input=ylabel_input,
                xlabel_input=kwargs.get("xlabel_input"),
            )
            p2 = function(
                data[1],
                "Actor-critic",
                fig.add_subplot(122),
                False,
                False,
                xlabel_input=kwargs.get("xlabel_input"),
            )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.show()
    plt.savefig(graph_path)


def prepare_reward_loss_plot(
    loss_data: [np.ndarray, tf.Tensor],
    reward_data: [np.ndarray, tf.Tensor],
    steps: int,
    title: str,
    xlabel: bool,
    axes: plt.axes,
) -> plt.plot:
    """Prepares a subplot with loss and reward data plotted to be included in a subplot
    :param loss_data: a series of loss data to be plotted
    :param reward_data: a series of reward data to be plotted
    :param steps: number of steps to make while calculating a moving average
    :param title: graph title
    :param xlabel: a boolean value if the X-axis label should be included
    :param axes: a subplot location
    :return: a subplot
    """
    loss_data = moving_average(loss_data, steps)
    reward_data = moving_average(reward_data, steps)
    plt = axes
    plt.plot(
        range(0, len(loss_data)), loss_data, linewidth=3, label="loss", color="tab:red"
    )
    plt.set_title(title, fontweight="bold")
    plt.set_ylabel("Moving average of episode loss", fontsize=12)
    if xlabel is True:
        plt.set_xlabel("No. of episodes", fontsize=12)

    ax2 = plt.twinx()
    if title == "Policy-based" or title == "Actor-critic":
        ax2.plot(
            range(0, len(reward_data)),
            reward_data,
            linewidth=3,
            label="reward\n(episode mean)",
            color="tab:green",
        )
    else:
        ax2.plot(
            range(0, len(reward_data)),
            reward_data,
            linewidth=3,
            label="reward",
            color="tab:green",
        )
    ax2.set_ylabel("Mving average of episode reward", fontsize=12)

    h1, l1 = plt.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2)
    return plt


def plot_loss_reward_subplots(
    loss_data: list, reward_data: list, steps: int, graph_path: str
):
    """Displays and saves a graph with two or four subplots
    :param loss_data: a series of loss data to be plotted
    :param reward_data: a series of reward data to be plotted
    :param steps: number of steps to make while calculating a moving average
    :param graph_path: path to save the graph
    """
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    grid_spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    p1 = prepare_reward_loss_plot(
        loss_data[0],
        reward_data[0],
        steps,
        "Nbandit",
        False,
        fig.add_subplot(grid_spec[0, 0]),
    )
    p2 = prepare_reward_loss_plot(
        loss_data[1],
        reward_data[1],
        steps,
        "Contextual bandit",
        False,
        fig.add_subplot(grid_spec[0, 1]),
    )
    p3 = prepare_reward_loss_plot(
        loss_data[2],
        reward_data[2],
        steps,
        "Policy-based",
        True,
        fig.add_subplot(grid_spec[1, 0]),
    )
    p4 = prepare_reward_loss_plot(
        loss_data[3],
        reward_data[3],
        steps,
        "Actor-critic",
        True,
        fig.add_subplot(grid_spec[1, 1]),
    )
    fig.suptitle(
        f"Moving averages (every {steps} steps) of episode losses and returns across all methods",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.savefig(graph_path)
