import os
import argparse
import tensorflow as tf

from model.model_fn import model_fn
from model.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/mnist-mobilenet',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':
    args = parser.parse_args()
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    export_dir = os.path.join(args.model_dir, "export")

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    tf.logging.info("Exporting the model...")
    images= tf.placeholder(tf.float32, [None, params.image_size, params.image_size, 3], name="images")

    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'images': images,
    })
    estimator.export_savedmodel(export_dir, input_fn)
    tf.logging.info("Done")
