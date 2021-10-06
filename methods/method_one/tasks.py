# Adding libraries required for test image preprocessing
import os
from datetime import datetime
import pytz
from luigi import Task
from luigi import IntParameter
from luigi import Parameter
from luigi import LocalTarget
from luigi import ListParameter
from luigi import FloatParameter
import pandas as pd
from methods.utils.other import Sampling
from methods.method_one.nbandit import nbandit
from methods.utils.other import save_pickle
from methods.utils.other import load_pickle
from methods.utils.visualizations import plot_loss
from methods.utils.visualizations import plot_weights
from methods.utils.visualizations import plot_weight_time


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
    base_read = os.path.join(base_dir, "data/metadata_tmp")
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
    """Class designed to pass Luigi parameters between tasks to avoid code repretition"""

    base_dir = Parameter(default="/home/kacper_krasowiak/")
    sample_size = IntParameter(default=5000)
    sampling_method = Parameter(default="no_filter")
    sampling_folder = IntParameter(default=None)
    sampling_focus_type = Parameter(default=None)
    description = Parameter(
        default=datetime.now(pytz.timezone("Europe/London")).strftime("%d_%m_%Y_%H_%M")
    )
    weights = ListParameter(default=[0.25, 0.25, 0.25, 0.25])
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


class RunNBandit(PassParameters, Task):
    """Luigi task to run the N-Bandit training and save the results in a specified path"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return PrepareSample(**self.collect_params())

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(os.path.split(self.input().path)[0], "results.pickle")
        )

    def run(self):
        """Runs the N-Bandit training and save the results in a specified path"""
        results = nbandit(pd.read_csv(self.input().path), self.weights, self.epsilon)
        save_pickle(results, self.output().path)


class PrepareVisualizations(PassParameters, Task):
    """Luigi task to create and save visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunNBandit(**self.collect_params())

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(os.path.split(self.input().path)[0], "visualizations")
        )

    def run(self):
        """Plots the results of the training procedure"""
        os.mkdir(os.path.join(os.path.split(self.input().path)[0], "visualizations"))
        results = load_pickle(self.input().path)
        plot_loss(results[0], os.path.join(self.output().path, "loss_plot.png"))
        plot_weights(
            results[1][-1], os.path.join(self.output().path, "weight_plot.png")
        )
        plot_weight_time(
            results[1], os.path.join(self.output().path, "weight_episodes_plot.png")
        )
