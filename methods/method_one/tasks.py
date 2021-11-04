import os
from luigi import Task
from luigi import LocalTarget
import pandas as pd
from methods.utils.shared_tasks import PassParameters, PrepareSample
from methods.method_one.nbandit import nbandit
from methods.utils.other import save_pickle
from methods.utils.shared_tasks import PrepareVisualizations


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
        results = nbandit(
            data=pd.read_csv(self.input().path),
            weights=self.weights,
            epsilon=self.epsilon,
            lr=self.lr,
        )
        save_pickle(results, self.output().path)


class PrepareVisualizationsNBandit(PrepareVisualizations):
    """Luigi task to create and save visualizations of the training results"""

    def requires(self):
        """Specified required preceding Luigi task"""
        return RunNBandit(**self.collect_params())
