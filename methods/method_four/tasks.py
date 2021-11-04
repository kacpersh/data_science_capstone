import os
from luigi import Task
from luigi import LocalTarget
import pandas as pd
from methods.utils.shared_tasks import PassParameters, PrepareSample
from methods.method_four.actor_critic import actor_critic
from methods.utils.other import save_pickle
from methods.utils.shared_tasks import PrepareComplexVisualizations


class RunAgentCritic(PassParameters, Task):
    """Luigi task to run the Agent-Critic method training and save the results in a specified path"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return PrepareSample(**self.collect_params())

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(os.path.split(self.input().path)[0], "results.pickle")
        )

    def run(self):
        """Runs the the Agent-Critic method training and save the results in a specified path"""
        results = actor_critic(
            data=pd.read_csv(self.input().path),
            epsilon=self.epsilon,
            max_steps=self.max_steps,
            gamma=self.gamma,
            lr=self.lr,
        )
        save_pickle(results, self.output().path)


class PrepareVisualizationsAgentCritic(PrepareComplexVisualizations):
    """Luigi task to create and save visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunAgentCritic(**self.collect_params())
