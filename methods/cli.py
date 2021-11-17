import argparse
from datetime import datetime
import pytz
from luigi import build
from methods.tasks import (
    PrepareAllVisualizations,
    PrepareSampleTuning,
    PrepareAllVisualizationsSample,
)

parser = argparse.ArgumentParser()
parser.add_argument("-bd", "--base_dir", default="/home/kacper_krasowiak/")
parser.add_argument("-ss", "--sample_size", default=5000, type=int)
parser.add_argument("-sm", "--sampling_method", default="no_filter")
parser.add_argument("-sf", "--sampling_folder", default=None)
parser.add_argument("-sft", "--sampling_focus_type", default=None)
parser.add_argument(
    "-d",
    "--description",
    default=datetime.now(pytz.timezone("Europe/London")).strftime("%d_%m_%Y_%H_%M"),
)
parser.add_argument("-e", "--epsilon", default=0.2, type=float)
parser.add_argument("-w", "--weights", nargs="*", default=[0.2, 0.2, 0.2, 0.2, 0.2])
parser.add_argument("-ms", "--max_steps", default=5, type=int)
parser.add_argument("-g", "--gamma", default=0.9, type=float)
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
parser.add_argument("-ps", "--prepare_sample", default=False, type=bool)
parser.add_argument("-sp", "--sample_path", default=None)
parser.add_argument("-en", "--experiment_name", default=None)
parser.add_argument("-mas", "--moving_average_steps", default=25, type=int)


def main(args=None):
    """Runs a Luigi task to create and save visualizations of the training results
    Parameterization via argparse:
    :param str base_dir: path to base directory where sample should be saved
    :param str sample_size: number of sample observations to be used during training
    :param str sampling_method: one of the Sampling class methods
    :param str sampling_folder: folder number if th Sampling class method requires
    :param str sampling_focus_type: focus type if th Sampling class method requires
    :param str description: description of the experiment
    :param str weights: a list of action weights
    :param str epsilon: probability bar to select an action different from the optimal one
    :param str max_steps: a maximum number of preprocessing steps the model can take on one image
    :param str gamma: a decaying discount factor, the higher the value the more forward looking the less weight for future values
    :param str learning_rate: learning rate for the Keras classification model
    :param str prepare_sample: boolean decision if the command goal is just to prepare a data sample for further experiments
    :param str sample_path: a path of a data sample for further experiments
    :param str experiment_name: name of the conducted experiment
    :param str moving_average_steps: number of steps to make while calculating a moving average
    :returns: save and print out the visualizations of the results after training
    """
    args = parser.parse_args()

    if args.prepare_sample is True:

        build(
            [
                PrepareSampleTuning(
                    base_dir=f"{args.base_dir}",
                    sample_size=args.sample_size,
                    sampling_method=f"{args.sampling_method}",
                    sampling_folder=f"{args.sampling_folder}",
                    sampling_focus_type=f"{args.sampling_focus_type}",
                    description=f"{args.description}",
                )
            ],
            local_scheduler=True,
        )

    else:

        if args.sample_path is not None:

            build(
                [
                    PrepareAllVisualizationsSample(
                        weights=args.weights,
                        epsilon=args.epsilon,
                        max_steps=args.max_steps,
                        gamma=args.gamma,
                        lr=args.learning_rate,
                        sample_path=f"{args.sample_path}",
                        experiment_name=f"{args.experiment_name}",
                        moving_average_steps=args.moving_average_steps,
                    )
                ],
                local_scheduler=True,
            )

        else:

            build(
                [
                    PrepareAllVisualizations(
                        base_dir=f"{args.base_dir}",
                        sample_size=args.sample_size,
                        sampling_method=f"{args.sampling_method}",
                        sampling_folder=f"{args.sampling_folder}",
                        sampling_focus_type=f"{args.sampling_focus_type}",
                        description=f"{args.description}",
                        weights=args.weights,
                        epsilon=args.epsilon,
                        max_steps=args.max_steps,
                        gamma=args.gamma,
                        lr=args.learning_rate,
                        moving_average_steps=args.moving_average_steps,
                    )
                ],
                local_scheduler=True,
            )


if __name__ == "__main__":
    main()
