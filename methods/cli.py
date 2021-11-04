import argparse
from datetime import datetime
import pytz
from luigi import build
from methods.tasks import PrepareAllVisualizations

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
    :returns: save and print out the visualizations of the results after training
    """
    args = parser.parse_args()
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
            )
        ],
        local_scheduler=True,
    )


if __name__ == "__main__":
    main()
