# Adding libraries required for test image preprocessing
import matplotlib.pyplot as plt


def plot_loss(
    loss: list,
    graph_path: str,
    xlabel: str = "No. of episodes",
    ylabel: str = "Custom loss",
    title: str = "Changes of custom loss against no. of episodes",
):
    """Displays and saves a graph showing changes of custom loss during the training process
    :param loss: list with loss after each episode
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param title: graph title
    :param graph_path: path to save the graph
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(loss)), loss, linewidth=3)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()
    plt.savefig(graph_path)


def plot_weights(
    data: list,
    graph_path: str,
    actions: list = [
        "scaleup_weight",
        "denoise_weight",
        "thresholding_weight",
        "brightness_weight",
    ],
    ylabel: str = "Action weight",
    title: str = "Action weights after training completion",
):
    """Displays and saves a bar plot of weights/rewards for each action
    :param data: values of weights/rewards for each action
    :param graph_path: path to save the graph
    :param actions: a list of functions used as actions in the training process
    :param ylabel: Y-axis label
    :param title: graph title
    """
    plt.figure(figsize=(20, 10))
    plt.bar(actions, data)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12)
    plt.show()
    plt.savefig(graph_path)


def plot_weight_time(
    data: list,
    graph_path: str,
    actions: list = [
        "scaleup_weight",
        "denoise_weight",
        "thresholding_weight",
        "brightness_weight",
    ],
    xlabel: str = "No. of episodes",
    ylabel: str = "Action weight",
    title: str = "Action weights against no. of episodes",
):
    """Displays and saves a plot of action weights during the training process
    :param data: a list of lists with action weights for each episode
    :param actions: a list of functions used as actions in the training process
    :param ylabel: Y-axis label
    :param title: graph title
    :param graph_path: path to save the graph
    """
    w1 = [i[0] for i in data]
    w2 = [i[1] for i in data]
    w3 = [i[2] for i in data]
    w4 = [i[3] for i in data]
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(w1)), w1, linewidth=3)
    plt.plot(range(0, len(w2)), w2, linewidth=3)
    plt.plot(range(0, len(w3)), w3, linewidth=3)
    plt.plot(range(0, len(w4)), w4, linewidth=3)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(actions)
    plt.show()
    plt.savefig(graph_path)
