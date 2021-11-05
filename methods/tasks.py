import os
import pandas as pd
from luigi import Task
from luigi import LocalTarget
from methods.utils.other import save_pickle, load_pickle
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
        """Runs the N-Bandit training and save the results in a specified path"""
        nbandit_results = nbandit(
            data=pd.read_csv(self.input().path),
            weights=self.weights,
            epsilon=self.epsilon,
            lr=self.lr,
        )
        cbandit_results = cbandit(
            data=pd.read_csv(self.input().path), epsilon=self.epsilon, lr=self.lr
        )
        pb_agent_results = pb_agent(
            data=pd.read_csv(self.input().path),
            epsilon=self.epsilon,
            max_steps=self.max_steps,
            gamma=self.gamma,
            lr=self.lr,
        )

        actor_critic_results = actor_critic(
            data=pd.read_csv(self.input().path),
            epsilon=self.epsilon,
            max_steps=self.max_steps,
            gamma=self.gamma,
            lr=self.lr,
        )

        save_pickle(
            [nbandit_results, cbandit_results, pb_agent_results, actor_critic_results],
            self.output().path,
        )


class PrepareAllVisualizations(PrepareSimpleVisualizations):
    """Luigi task to create and save all visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunAll(**self.collect_params())

    def run(self):
        """Plots the results of the training procedure"""
        os.mkdir(os.path.join(os.path.split(self.input().path)[0], "visualizations"))
        results = load_pickle(self.input().path)

        loss = [i[0] for i in results]
        running_cumulative_reward = [i[2] for i in results]
        running_episode_duration = [i[3] for i in results]
        running_reward = [i[1] for i in results]
        running_reward = [
            [float(i.numpy()) for i in running_reward[0]],
            [float(i.numpy()) for i in running_reward[1]],
            [float(i[0].numpy()) for i in running_reward[2]],
            [float(i[0].numpy()) for i in running_reward[3]],
        ]
        running_actions_count = [i[4] for i in results]
        running_step_count = [i[5] for i in results[-2:]]
        running_actions_combinations_count = [i[6] for i in results[-2:]]

        plot_summary_series(
            standardizer(loss),
            "Changes of standardized loss over no. of episodes across all methods",
            graph_path=os.path.join(self.output().path, "running_loss_plot.png"),
            ylabel="Standardized loss",
        )
        plot_summary_series(
            running_cumulative_reward,
            "Changes of cumulative reward over no. of episodes across all methods",
            graph_path=os.path.join(
                self.output().path, "running_cumulative_episode_reward.png"
            ),
            ylabel="Cumulative reward",
        )
        plot_summary_series(
            running_episode_duration,
            "Changes of episode duration over no. of episodes across all methods",
            graph_path=os.path.join(self.output().path, "running_episode_duration.png"),
            ylabel="Episode duration [s]",
        )
        plot_subplots(
            running_reward,
            True,
            "Histograms of episode rewards across all methods",
            prepare_histogram,
            "Count",
            os.path.join(self.output().path, "running_histogram_episode_reward.png"),
            xlabel_input="Reward",
        )
        plot_subplots(
            running_actions_count,
            True,
            "Changes of cumulative action count over no. of episodes across all methods",
            prepare_cumulative_action_count,
            "Cumulative action count",
            os.path.join(
                self.output().path, "running_cumulative_episode_actions_count.png"
            ),
            xlabel_input="No. of episodes",
        )
        plot_subplots(
            running_step_count,
            False,
            "Changes of no. of steps per episode over no. of episodes \n for the policy-based and action critic methods",
            prepare_histogram,
            "No. of steps per episode",
            os.path.join(self.output().path, "running_step_count.png"),
            combinations=False,
            xlabel_input="No. of episodes",
        )
        plot_subplots(
            running_actions_combinations_count,
            False,
            "Count of top 5 unique action combinations from the training session \n for the policy-based and action critic methods",
            prepare_cumulative_combination_count,
            "Count of unique action combinations",
            os.path.join(self.output().path, "actions_combinations_count.png"),
            combinations=True,
        )
