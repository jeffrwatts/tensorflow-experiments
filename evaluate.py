"""Train the model"""

import argparse
import os

import tensorflow as tf

from model.input_fn import input_fn, get_validation_filenames
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/mnist-lenet',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Create the input data pipeline
    tf.logging.info("Creating the datasets...")
    dataset_dir = os.path.join(args.data_dir, params.dataset)
    validation_data_filenames = get_validation_filenames(dataset_dir, params.dataset)
    print(validation_data_filenames)

    # Create the two input functions over the two datasets
    validate_input_fn = lambda: input_fn(is_training=False, data_filenames=validation_data_filenames, params=params)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(validate_input_fn)
    for key in res:
        print("{}: {}".format(key, res[key]))
