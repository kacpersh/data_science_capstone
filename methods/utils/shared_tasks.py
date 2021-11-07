import os
from datetime import datetime
import pytz
from luigi import IntParameter
from luigi import Parameter
from luigi import ListParameter
from luigi import FloatParameter
from luigi import Task
from luigi import LocalTarget
from methods.utils.other import Sampling
from methods.utils.other import load_pickle
from methods.utils.visualizations import (
    plot_time_series,
    plot_cumulative_action_count,
    plot_cumulative_combination_count,
)


def luigi_sample_runner(
    base_dir: str,
    description: str,
    sample_size: int,
    sampling_method: str,
    sampling_folder: int,
    sampling_focus_type: str,
    output: str,
    hyper_tuning: bool = False,
):
    """Executes the sampling procedure with the Sampling class and specified method
    :param base_dir: path to base directory where sample should be saved
    :param description: description of the experiment
    :param sample_size: number of sample observations to be used during training
    :param sampling_method: one of the Sampling class methods
    :param sampling_folder: folder number if th Sampling class method requires
    :param sampling_focus_type: focus type if th Sampling class method requires
    :param output: path to save the sample
    :param hyper_tuning: boolean if the sample is meant for hyperparamter tuning experiments
    """
    base_output = os.path.join(base_dir, "output")
    base_read = os.path.join(base_dir, "data/metadata")
    if os.path.exists(base_output) is False:
        os.mkdir(base_output)
        os.system("sudo chmod 777 " + os.path.join(base_dir, "output"))
    if hyper_tuning is True:
        os.mkdir(base_output + "/tuning_experiments_sample_" + description)
    else:
        os.mkdir(base_output + "/experiment_" + description)
    sample_class = Sampling(base_read, sample_size)
    method = getattr(sample_class, sampling_method)
    if sampling_method == "filter_a":
        sample = method(sampling_folder)
    elif sampling_method == "filter_b":
        sample = method(sampling_focus_type)
    elif sampling_method == "filter_ab":
        sample = method(sampling_folder, sampling_focus_type)
    else:
        sample = method()
    sample.to_csv(output)


class PassParameters(object):
    """Class designed to pass Luigi parameters between tasks to avoid code repetition"""

    base_dir = Parameter(default="/home/kacper_krasowiak/")
    sample_size = IntParameter(default=5000)
    sampling_method = Parameter(default="no_filter")
    sampling_folder = IntParameter(default=None)
    sampling_focus_type = Parameter(default=None)
    description = Parameter(
        default=datetime.now(pytz.timezone("Europe/London")).strftime("%d_%m_%Y_%H_%M")
    )
    weights = ListParameter(default=[0.2, 0.2, 0.2, 0.2, 0.2])
    epsilon = FloatParameter(default=0.2)
    max_steps = IntParameter(default=5)
    gamma = FloatParameter(default=0.9)
    lr = FloatParameter(default=0.001)
    sample_path = Parameter(default=None)
    experiment_name = Parameter(default=None)
    loss_sampling_steps = IntParameter(default=25)

    def collect_params(self):
        return {
            "base_dir": self.base_dir,
            "sample_size": self.sample_size,
            "sampling_method": self.sampling_method,
            "sampling_folder": self.sampling_folder,
            "sampling_focus_type": self.sampling_focus_type,
            "description": self.description,
            "weights": self.weights,
            "epsilon": self.epsilon,
            "max_steps": self.max_steps,
            "gamma": self.gamma,
            "lr": self.lr,
            "sample_path": self.sample_path,
            "experiment_name": self.experiment_name,
            "loss_sampling_steps": self.loss_sampling_steps,
        }


class PrepareSample(PassParameters, Task):
    """Luigi task to read a dataframe with image metadata and to generate a random sample of specified size and properties"""

    def requires(self):
        return []

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(
                self.base_dir,
                "output",
                "experiment_" + self.description,
                ("sample_metadata.csv"),
            )
        )

    def run(self):
        """Reads a dataframe with image metadata and generates a random sample of specified size and properties"""
        luigi_sample_runner(
            self.base_dir,
            self.description,
            self.sample_size,
            self.sampling_method,
            self.sampling_folder,
            self.sampling_focus_type,
            self.output().path,
        )


class PrepareSimpleVisualizations(PassParameters, Task):
    """Luigi task to create and save visualizations of the training results"""

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(os.path.split(self.input().path)[0], "visualizations")
        )

    def run(self):
        """Plots the results of the training procedure"""
        os.mkdir(os.path.join(os.path.split(self.input().path)[0], "visualizations"))
        results = load_pickle(self.input().path)
        loss_series = results[0]
        running_total_episode_reward = results[1]
        running_cumulative_episode_reward = results[2]
        running_episode_duration = results[3]
        running_cumulative_episode_actions_count = results[4]

        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "loss_plot.png"),
            "Loss",
            "Changes of loss over no. of episodes",
        )
        plot_time_series(
            running_total_episode_reward,
            os.path.join(self.output().path, "running_total_episode_reward.png"),
            "Reward value",
            "Changes of episode reward values over no. of episodes",
        )
        plot_time_series(
            running_cumulative_episode_reward,
            os.path.join(self.output().path, "running_cumulative_episode_reward.png"),
            "Cumulative reward value",
            "Changes of cumulative reward over no. of episodes",
        )
        plot_time_series(
            running_episode_duration,
            os.path.join(self.output().path, "running_episode_duration.png"),
            "Episode duration [s]",
            "Changes of episode duration over no. of episodes",
        )
        plot_cumulative_action_count(
            running_cumulative_episode_actions_count,
            os.path.join(
                self.output().path, "running_cumulative_episode_actions_count.png"
            ),
        )


class PrepareComplexVisualizations(PrepareSimpleVisualizations):
    """Luigi task to create and save visualizations of the training results"""

    def run(self):
        """Plots the results of the training procedure"""
        os.mkdir(os.path.join(os.path.split(self.input().path)[0], "visualizations"))
        results = load_pickle(self.input().path)
        loss_series = results[0]
        running_total_episode_reward = results[1]
        running_cumulative_episode_reward = results[2]
        running_episode_duration = results[3]
        running_cumulative_episode_actions_count = results[4]
        running_step_count = results[5]
        actions_combinations_count = results[6]

        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "loss_plot.png"),
            "Loss",
            "Changes of loss over no. of episodes",
        )
        plot_time_series(
            running_step_count,
            os.path.join(self.output().path, "running_step_count.png"),
            "No. of steps per episode",
            "Changes of no. of steps per episode over no. of episodes",
        )
        plot_time_series(
            running_total_episode_reward,
            os.path.join(self.output().path, "running_total_episode_reward.png"),
            "Reward value",
            "Changes of episode reward values over no. of episodes",
        )
        plot_time_series(
            running_cumulative_episode_reward,
            os.path.join(self.output().path, "running_cumulative_episode_reward.png"),
            "Cumulative reward value",
            "Changes of cumulative reward over no. of episodes",
        )
        plot_time_series(
            running_episode_duration,
            os.path.join(self.output().path, "running_episode_duration.png"),
            "Episode duration [s]",
            "Changes of episode duration over no. of episodes",
        )
        plot_cumulative_action_count(
            running_cumulative_episode_actions_count,
            os.path.join(
                self.output().path, "running_cumulative_episode_actions_count.png"
            ),
        )
        plot_cumulative_combination_count(
            actions_combinations_count,
            os.path.join(self.output().path, "actions_combinations_count.png"),
        )
