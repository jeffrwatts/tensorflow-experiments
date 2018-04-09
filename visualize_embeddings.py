"""Train the model"""

import argparse
import os
import imageio
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model.utils import Params
from model.utils import images_to_sprite
from model.model_fn import model_fn
from model.input_fn import raw_dataset, get_validation_filenames


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/mnist-mobilenet',
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
    tf.logging.info("Extracting the dataset...")
    dataset_dir = os.path.join(args.data_dir, params.dataset)
    validation_data_filenames = get_validation_filenames(dataset_dir, params.dataset)
    print(validation_data_filenames)

    # Create the two input functions over the two datasets
    validate_input_fn = lambda: raw_dataset(include_raw_image=True, data_filenames=validation_data_filenames, params=params)

    # A bit wasteful with memory having both raw and normalized image at once... revisit this.
    sprint_image_size = params.image_size
    if sprint_image_size > 64:
        sprint_image_size = 64

    test_index = 0
    test_images = np.zeros((params.validate_size, sprint_image_size, sprint_image_size, 3), dtype=np.uint8)
    test_normalized_images = np.zeros((params.validate_size, params.image_size, params.image_size, 3), dtype=np.float32)
    test_labels = np.zeros((params.validate_size), dtype=int)

    with tf.Session() as sess:
        iterator = validate_input_fn().make_one_shot_iterator()
        next_element = iterator.get_next()

        while True:
            try:
                image, image_normalized, label = sess.run(next_element)

                for batchIx in range(image.shape[0]):
                    if sprint_image_size != params.image_size:
                        test_images[test_index] = cv2.resize(image[batchIx], (sprint_image_size, sprint_image_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        test_images[test_index] = image[batchIx]
                    test_normalized_images[test_index,:,:,:] = image_normalized[batchIx]
                    test_labels[test_index] = label[batchIx]
                    test_index+=1

            except tf.errors.OutOfRangeError:
                break

    # Use raw image to generate the sprite.
    tf.logging.info("Creating sprite...")
    eval_dir = os.path.join(args.model_dir, "eval")
    sprite_image = images_to_sprite(test_images)
    imageio.imwrite(os.path.join(eval_dir, 'sprite.png'), sprite_image)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)


    # Compute embeddings on the test set
    tf.logging.info("Predicting...")

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(test_normalized_images, num_epochs=1,
                                                          batch_size=params.batch_size, shuffle=False)
    predictions = estimator.predict(predict_input_fn)

    embeddings = np.zeros((params.validate_size, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

    # Visualize test embeddings
    embedding_var = tf.Variable(embeddings, name='embedding_var')

    summary_writer = tf.summary.FileWriter(eval_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    # Save the metadata file needed for Tensorboard projector
    metadata_filename = "embedding_metadata.tsv"
    with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        for i in range(params.validate_size):
            c = test_labels[i]
            f.write('{}\n'.format(c))
    embedding.metadata_path = metadata_filename

    # Specify where you find the sprite (we will create this later)
    # Copy the embedding sprite image to the eval directory
    embedding.sprite.image_path = "sprite.png"
    embedding.sprite.single_image_dim.extend([sprint_image_size, sprint_image_size])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))
