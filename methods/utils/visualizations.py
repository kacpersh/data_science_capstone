import matplotlib.pyplot as plt


def plot_time_series(
    data: list,
    graph_path: str,
    ylabel: str,
    title: str,
    xlabel: str = "No. of episodes",
):
    """Displays and saves a graph showing changes of custom loss during the training process
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
        "scaleup_weight",
        "denoise_weight",
        "thresholding_weight",
        "brightness_weight",
        "no_action_weight",
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
