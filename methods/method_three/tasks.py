import os
from luigi import Task
from luigi import LocalTarget
import pandas as pd
from methods.utils.visualizations import (
    plot_time_series,
    plot_cumulative_action_count,
    plot_cumulative_combination_count,
)
from methods.utils.shared_tasks import PassParameters, PrepareSample
from methods.method_three.pb_agent import pb_agent
from methods.utils.other import save_pickle, load_pickle
from methods.utils.shared_tasks import PrepareVisualizations


class RunPbAgent(PassParameters, Task):
    """Luigi task to run the Policy-based Agent training and save the results in a specified path"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return PrepareSample(**self.collect_params())

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(os.path.split(self.input().path)[0], "results.pickle")
        )

    def run(self):
        """Runs the the Policy-based Agent training and save the results in a specified path"""
        results = pb_agent(
            data=pd.read_csv(self.input().path),
            epsilon=self.epsilon,
            max_steps=self.max_steps,
            gamma=self.gamma,
            lr=self.lr,
        )
        save_pickle(results, self.output().path)


class PrepareVisualizationsPbAgent(PrepareVisualizations):
    """Luigi task to create and save visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunPbAgent(**self.collect_params())

    def run(self):
        """Plots the results of the training procedure"""
        os.mkdir(os.path.join(os.path.split(self.input().path)[0], "visualizations"))
        results = load_pickle(self.input().path)
        loss_series = results[0]
        running_step_count = results[1]
        running_total_episode_reward = results[2]
        running_cumulative_episode_reward = results[3]
        running_episode_duration = results[4]
        running_cumulative_episode_actions_count = results[5]
        actions_combinations_count = results[6]

        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "loss_plot.png"),
            "Loss",
            "Changes of custom loss over no. of episodes",
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
