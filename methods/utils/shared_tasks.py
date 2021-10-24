import os
from datetime import datetime
import pytz
from statistics import mean
from luigi import IntParameter
from luigi import Parameter
from luigi import ListParameter
from luigi import FloatParameter
from luigi import Task
from luigi import LocalTarget
from methods.utils.other import Sampling
from methods.utils.other import load_pickle
from methods.utils.rewards import cumulative_rewards
from methods.utils.visualizations import plot_time_series
from methods.utils.visualizations import plot_cumulative_action_count


def luigi_sample_runner(
    base_dir: str,
    description: str,
    sample_size: int,
    sampling_method: str,
    sampling_folder: int,
    sampling_focus_type: str,
    output: str,
):
    """Executes the sampling procedure with the Sampling class and specified method
    :param base_dir: path to base directory where sample should be saved
    :param description: description of the experiment
    :param sample_size: number of sample observations to be used during training
    :param sampling_method: one of the Sampling class methods
    :param sampling_folder: folder number if th Sampling class method requires
    :param sampling_focus_type: focus type if th Sampling class method requires
    :param output: path to save the sample
    """
    base_output = os.path.join(base_dir, "output")
    base_read = os.path.join(base_dir, "data/metadata")
    if os.path.exists(base_output) is False:
        os.mkdir(base_output)
        os.system("sudo chmod 777" + os.path.join(base_dir, "output"))
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


class PrepareVisualizations(PassParameters, Task):
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
        cumulative_reward_series = cumulative_rewards(results[1])
        action_cumulative_count = results[2]
        duration_series = results[3]
        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "loss_plot.png"),
            "Custom loss",
            "Changes of custom loss over no. of episodes",
        )
        plot_time_series(
            cumulative_reward_series,
            os.path.join(self.output().path, "cumulative_reward_plot.png"),
            "Cumulative reward",
            "Changes of cumulative reward over no. of episodes",
        )
        plot_cumulative_action_count(
            action_cumulative_count,
            os.path.join(self.output().path, "cumulative_action_count_plot.png"),
        )
        plot_time_series(
            duration_series,
            os.path.join(self.output().path, "episode_duration_plot.png"),
            "Episode duration [s]",
            f"Changes of episode duration over no. of episodes \n mean duration={round(mean(duration_series), 2)}",
        )
