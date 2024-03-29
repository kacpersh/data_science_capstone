import os
from luigi import Task
from luigi import LocalTarget
import pandas as pd
from methods.utils.shared_tasks import PassParameters, PrepareSample
from methods.method_two.cbandit import cbandit
from methods.utils.other import save_pickle
from methods.utils.shared_tasks import PrepareSimpleVisualizations


class RunCBandit(PassParameters, Task):
    """Luigi task to run the Contextual Bandit training and save the results in a specified path"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return PrepareSample(**self.collect_params())

    def output(self):
        """Target object path"""
        return LocalTarget(
            os.path.join(os.path.split(self.input().path)[0], "results.pickle")
        )

    def run(self):
        """Runs the Contextual Bandit training and save the results in a specified path"""
        results = cbandit(
            data=pd.read_csv(self.input().path), epsilon=self.epsilon, lr=self.lr
        )
        save_pickle(results, self.output().path)


class PrepareVisualizationsCBandit(PrepareSimpleVisualizations):
    """Luigi task to create and save visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunCBandit(**self.collect_params())
