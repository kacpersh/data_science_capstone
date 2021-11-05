import os
import pandas as pd
from luigi import Task
from luigi import LocalTarget
from methods.utils.other import save_pickle, load_pickle
from methods.utils.shared_tasks import luigi_sample_runner
from methods.utils.shared_tasks import PassParameters, PrepareSample
from methods.method_one.nbandit import nbandit
from methods.method_two.cbandit import cbandit
from methods.method_three.pb_agent import pb_agent
from methods.method_four.actor_critic import actor_critic
from methods.utils.rewards import standardizer
from methods.utils.shared_tasks import PrepareSimpleVisualizations
from methods.utils.visualizations import (
    plot_summary_series,
    plot_subplots,
    prepare_histogram,
    prepare_cumulative_action_count,
    prepare_cumulative_combination_count,
)


def run_all_methods(input, weights, epsilon, lr, max_steps, gamma, output):
    """Support function for a Luigi task to execute all methods from a shared data sample
    :param input: path to a Pandas dataframe with a training sample
    :param weights:  a list of action weights
    :param epsilon: probability bar to select an action different from the optimal one
    :param lr: learning rate for the Keras classification model
    :param max_steps: a maximum number of preprocessing steps the model can take on one image
    :param gamma: a decaying discount factor, the higher the value the more forward looking the less weight for future values
    :param output: path to save the pickled file with outputs
    """
    nbandit_results = nbandit(
        data=pd.read_csv(input),
        weights=weights,
        epsilon=epsilon,
        lr=lr,
    )
    cbandit_results = cbandit(data=pd.read_csv(input), epsilon=epsilon, lr=lr)
    pb_agent_results = pb_agent(
        data=pd.read_csv(input),
        epsilon=epsilon,
        max_steps=max_steps,
        gamma=gamma,
        lr=lr,
    )

    actor_critic_results = actor_critic(
        data=pd.read_csv(input),
        epsilon=epsilon,
        max_steps=max_steps,
        gamma=gamma,
        lr=lr,
    )

    save_pickle(
        [nbandit_results, cbandit_results, pb_agent_results, actor_critic_results],
        output,
    )


def make_all_plots(input, output):
    """Support function for a Luigi task to print and save summary plots
    :param input: path to a pickled file with training results
    :param output: path to a folder where the summary plots should be saved
    """
    loss = [i[0] for i in input]
    running_cumulative_reward = [i[2] for i in input]
    running_episode_duration = [i[3] for i in input]
    running_reward = [i[1] for i in input]
    running_reward = [
        [float(i.numpy()) for i in running_reward[0]],
        [float(i.numpy()) for i in running_reward[1]],
        [float(i[0].numpy()) for i in running_reward[2]],
        [float(i[0].numpy()) for i in running_reward[3]],
    ]
    running_actions_count = [i[4] for i in input]
    running_step_count = [i[5] for i in input[-2:]]
    running_actions_combinations_count = [i[6] for i in input[-2:]]

    plot_summary_series(
        standardizer(loss),
        "Changes of standardized loss over no. of episodes across all methods",
        graph_path=os.path.join(output, "running_loss_plot.png"),
        ylabel="Standardized loss",
    )
    plot_summary_series(
        running_cumulative_reward,
        "Changes of cumulative reward over no. of episodes across all methods",
        graph_path=os.path.join(output, "running_cumulative_episode_reward.png"),
        ylabel="Cumulative reward",
    )
    plot_summary_series(
        running_episode_duration,
        "Changes of episode duration over no. of episodes across all methods",
        graph_path=os.path.join(output, "running_episode_duration.png"),
        ylabel="Episode duration [s]",
    )
    plot_subplots(
        running_reward,
        True,
        "Histograms of episode rewards across all methods",
        prepare_histogram,
        "Count",
        os.path.join(output, "running_histogram_episode_reward.png"),
        xlabel_input="Reward",
    )
    plot_subplots(
        running_actions_count,
        True,
        "Changes of cumulative action count over no. of episodes across all methods",
        prepare_cumulative_action_count,
        "Cumulative action count",
        os.path.join(output, "running_cumulative_episode_actions_count.png"),
        xlabel_input="No. of episodes",
    )
    plot_subplots(
        running_step_count,
        False,
        "Changes of no. of steps per episode over no. of episodes \n for the policy-based and action critic methods",
        prepare_histogram,
        "No. of steps per episode",
        os.path.join(output, "running_step_count.png"),
        combinations=False,
        xlabel_input="No. of episodes",
    )
    plot_subplots(
        running_actions_combinations_count,
        False,
        "Count of top 5 unique action combinations from the training session \n for the policy-based and action critic methods",
        prepare_cumulative_combination_count,
        "Count of unique action combinations",
        os.path.join(output, "actions_combinations_count.png"),
        combinations=True,
    )


class RunAll(PassParameters, Task):
    """Luigi task to run training for all the experiments and save the results in a specified path"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return PrepareSample(**self.collect_params())

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(os.path.split(self.input().path)[0], "results.pickle")
        )

    def run(self):
        """Runs the training and save the results in a specified path"""
        input = self.input().path
        weights = self.weights
        epsilon = self.epsilon
        lr = self.lr
        max_steps = self.max_steps
        gamma = self.gamma
        output = self.output().path

        run_all_methods(input, weights, epsilon, lr, max_steps, gamma, output)


class PrepareAllVisualizations(PrepareSimpleVisualizations):
    """Luigi task to create and save all visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunAll(**self.collect_params())

    def run(self):
        """Plots the results of the training procedure"""
        output = os.path.join(os.path.split(self.input().path)[0], "visualizations")
        if os.path.exists(output) is False:
            os.mkdir(output)
            os.system("sudo chmod 777 " + output)
        input = load_pickle(self.input().path)

        make_all_plots(input, output)


class PrepareSampleTuning(PrepareSample):
    """Luigi task to read a dataframe with image metadata and to generate a random sample of specified size and properties for hyper parameter tuning"""

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(
                self.base_dir,
                "output",
                "tuning_experiments_sample_" + self.description,
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
            True,
        )


class SampleExists(PassParameters, Task):
    """Luigi task to check if a sample of a provided path exists"""

    def requires(self):
        pass

    def output(self):
        """Target object path"""
        return LocalTarget(self.sample_path)


class RunAllSample(RunAll):
    """Luigi task to run training for all the experiments and save the results in a specified path"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return SampleExists(**self.collect_params())

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(
                os.path.split(self.input().path)[0],
                self.experiment_name,
                "results.pickle",
            )
        )

    def run(self):
        """Runs the training and save the results in a specified path"""
        output_path = os.path.join(
            os.path.split(self.input().path)[0], self.experiment_name
        )
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)
            os.system("sudo chmod 777 " + output_path)
        input = self.input().path
        weights = self.weights
        epsilon = self.epsilon
        lr = self.lr
        max_steps = self.max_steps
        gamma = self.gamma
        output = self.output().path

        run_all_methods(input, weights, epsilon, lr, max_steps, gamma, output)


class PrepareAllVisualizationsSample(PrepareSimpleVisualizations):
    """Luigi task to create and save all visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunAllSample(**self.collect_params())

    def run(self):
        """Plots the results of the training procedure"""
        output = os.path.join(os.path.split(self.input().path)[0], "visualizations")
        if os.path.exists(output) is False:
            os.mkdir(output)
            os.system("sudo chmod 777 " + output)
        input = load_pickle(self.input().path)

        make_all_plots(input, output)
