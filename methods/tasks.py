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
from methods.utils.visualizations import (
    plot_time_series,
    plot_cumulative_action_count,
    plot_cumulative_combination_count,
)
from methods.utils.shared_tasks import PrepareSimpleVisualizations


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

        # N-bandit results
        loss_series = results[0][0]
        running_total_episode_reward = results[0][1]
        running_cumulative_episode_reward = results[0][2]
        running_episode_duration = results[0][3]
        running_cumulative_episode_actions_count = results[0][4]

        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "nb_loss_plot.png"),
            "Loss",
            "N-BANDIT - Changes of custom loss over no. of episodes",
        )
        plot_time_series(
            running_total_episode_reward,
            os.path.join(self.output().path, "nb_running_total_episode_reward.png"),
            "Reward value",
            "N-BANDIT - Changes of episode reward values over no. of episodes",
        )
        plot_time_series(
            running_cumulative_episode_reward,
            os.path.join(
                self.output().path, "nb_running_cumulative_episode_reward.png"
            ),
            "Cumulative reward value",
            "N-BANDIT - Changes of cumulative reward over no. of episodes",
        )
        plot_time_series(
            running_episode_duration,
            os.path.join(self.output().path, "nb_running_episode_duration.png"),
            "Episode duration [s]",
            "N-BANDIT - Changes of episode duration over no. of episodes",
        )
        plot_cumulative_action_count(
            running_cumulative_episode_actions_count,
            os.path.join(
                self.output().path, "nb_running_cumulative_episode_actions_count.png"
            ),
            title="N-BANDIT - Changes of cumulative action count over no. of episodes",
        )

        # Contextual bandit results
        loss_series = results[1][0]
        running_total_episode_reward = results[1][1]
        running_cumulative_episode_reward = results[1][2]
        running_episode_duration = results[1][3]
        running_cumulative_episode_actions_count = results[1][4]

        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "cb_loss_plot.png"),
            "Loss",
            "CONTEXTUAL BANDIT - Changes of loss over no. of episodes",
        )
        plot_time_series(
            running_total_episode_reward,
            os.path.join(self.output().path, "cb_running_total_episode_reward.png"),
            "Reward value",
            "CONTEXTUAL BANDIT - Changes of episode reward values over no. of episodes",
        )
        plot_time_series(
            running_cumulative_episode_reward,
            os.path.join(
                self.output().path, "cb_running_cumulative_episode_reward.png"
            ),
            "Cumulative reward value",
            "CONTEXTUAL BANDIT - Changes of cumulative reward over no. of episodes",
        )
        plot_time_series(
            running_episode_duration,
            os.path.join(self.output().path, "cb_running_episode_duration.png"),
            "Episode duration [s]",
            "CONTEXTUAL BANDIT - Changes of episode duration over no. of episodes",
        )
        plot_cumulative_action_count(
            running_cumulative_episode_actions_count,
            os.path.join(
                self.output().path, "cb_running_cumulative_episode_actions_count.png"
            ),
            title="CONTEXTUAL BANDIT - Changes of cumulative action count over no. of episodes",
        )

        # Policy-based results
        loss_series = results[2][0]
        running_step_count = results[2][1]
        running_total_episode_reward = results[2][2]
        running_cumulative_episode_reward = results[2][3]
        running_episode_duration = results[2][4]
        running_cumulative_episode_actions_count = results[2][5]
        actions_combinations_count = results[2][6]

        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "pb_loss_plot.png"),
            "Loss",
            "POLICY-BASED AGENT - Changes of loss over no. of episodes",
        )
        plot_time_series(
            running_step_count,
            os.path.join(self.output().path, "pb_running_step_count.png"),
            "No. of steps per episode",
            "POLICY-BASED AGENT - Changes of no. of steps per episode over no. of episodes",
        )
        plot_time_series(
            running_total_episode_reward,
            os.path.join(self.output().path, "pb_running_total_episode_reward.png"),
            "Reward value",
            "POLICY-BASED AGENT - Changes of episode reward values over no. of episodes",
        )
        plot_time_series(
            running_cumulative_episode_reward,
            os.path.join(
                self.output().path, "pb_running_cumulative_episode_reward.png"
            ),
            "Cumulative reward value",
            "POLICY-BASED AGENT - Changes of cumulative reward over no. of episodes",
        )
        plot_time_series(
            running_episode_duration,
            os.path.join(self.output().path, "pb_running_episode_duration.png"),
            "Episode duration [s]",
            "POLICY-BASED AGENT - Changes of episode duration over no. of episodes",
        )
        plot_cumulative_action_count(
            running_cumulative_episode_actions_count,
            os.path.join(
                self.output().path, "pb_running_cumulative_episode_actions_count.png"
            ),
            title="POLICY-BASED AGENT - Changes of cumulative action count over no. of episodes",
        )
        plot_cumulative_combination_count(
            actions_combinations_count,
            os.path.join(self.output().path, "pb_actions_combinations_count.png"),
            title="POLICY-BASED AGENT - Count of top 5 unique action combinations from the training session",
        )

        # Actor-critic results
        loss_series = results[3][0]
        running_step_count = results[3][1]
        running_total_episode_reward = results[3][2]
        running_cumulative_episode_reward = results[3][3]
        running_episode_duration = results[3][4]
        running_cumulative_episode_actions_count = results[3][5]
        actions_combinations_count = results[3][6]

        plot_time_series(
            loss_series,
            os.path.join(self.output().path, "ac_loss_plot.png"),
            "Loss",
            "ACTOR-CRITIC - Changes of loss over no. of episodes",
        )
        plot_time_series(
            running_step_count,
            os.path.join(self.output().path, "ac_running_step_count.png"),
            "No. of steps per episode",
            "ACTOR-CRITIC - Changes of no. of steps per episode over no. of episodes",
        )
        plot_time_series(
            running_total_episode_reward,
            os.path.join(self.output().path, "ac_running_total_episode_reward.png"),
            "Reward value",
            "ACTOR-CRITIC - Changes of episode reward values over no. of episodes",
        )
        plot_time_series(
            running_cumulative_episode_reward,
            os.path.join(
                self.output().path, "ac_running_cumulative_episode_reward.png"
            ),
            "Cumulative reward value",
            "ACTOR-CRITIC - Changes of cumulative reward over no. of episodes",
        )
        plot_time_series(
            running_episode_duration,
            os.path.join(self.output().path, "ac_running_episode_duration.png"),
            "Episode duration [s]",
            "ACTOR-CRITIC - Changes of episode duration over no. of episodes",
        )
        plot_cumulative_action_count(
            running_cumulative_episode_actions_count,
            os.path.join(
                self.output().path, "ac_running_cumulative_episode_actions_count.png"
            ),
            title="ACTOR-CRITIC - Changes of cumulative action count over no. of episodes",
        )
        plot_cumulative_combination_count(
            actions_combinations_count,
            os.path.join(self.output().path, "ac_actions_combinations_count.png"),
            title="ACTOR-CRITIC - Count of top 5 unique action combinations from the training session",
        )
