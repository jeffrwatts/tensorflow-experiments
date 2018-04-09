import sys
import tensorflow as tf
from tensorflow.contrib import slim

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss

sys.path.append("../models/research/slim")
from nets import mobilenet_v1
from nets import lenet

def _build_mobilenet_model(is_training, images, params):
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)):
        out, _ = mobilenet_v1.mobilenet_v1(
            images,
            is_training=is_training,
            depth_multiplier=params.depth_multiplier,
            num_classes=None)
        tf.logging.info("mobilenet preembedding shape{}".format(out.get_shape().as_list()))
        out = tf.reshape(out, [-1, 256])
        out = tf.layers.dense(out, params.embedding_size, name="embeddings")
    return out

def _build_lenet_model(is_training, images, params):
    with slim.arg_scope(lenet.lenet_arg_scope()):
        out, _ = lenet.lenet(images, num_classes=None, is_training=is_training)
        tf.logging.info("lenet preembedding shape{}".format(out.get_shape().as_list()))
        #out = tf.reshape(out, [-1, 1024])
        out = tf.layers.dense(out, params.embedding_size, name="embeddings")
    return out

def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    images = features
    if isinstance(images, dict):
        images = features['images']

    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        if params.model == "mobilenet":
            tf.logging.info("Using mobilenet")
            embeddings = _build_mobilenet_model(is_training, images, params)
        elif params.model == "lenet":
            tf.logging.info("Using lenet")
            embeddings = _build_lenet_model(is_training, images, params)
        else:
            raise ValueError("model: {}".format(params.model))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode,
            predictions=predictions,
            export_outputs={
                'embeddings': tf.estimator.export.PredictOutput(predictions)
            })

    labels = tf.cast(labels, tf.int64)
    labels = tf.reshape(labels, [-1])

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    elif params.triplet_strategy == "tensorflow_semihard":
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings, margin=params.margin)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = dict()

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    tf.summary.image('train_image', images, max_outputs=1)

    global_step = tf.train.get_global_step()
    decay_steps = int(params.train_size / params.batch_size * params.num_epochs_per_decay)
    tf.logging.info("decay_steps={}".format(decay_steps))

    learning_rate = tf.train.exponential_decay(
        learning_rate=params.learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=params.decay_rate,
        staircase=True)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    tf.summary.scalar('learning_rate', learning_rate)
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
