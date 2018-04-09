import os
import tensorflow as tf

def _parse_example(example_proto, params, include_raw_image):
    features = {
          'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
          'image/format': tf.FixedLenFeature((), tf.string, default_value=params.image_format),
          'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    if params.image_format=="jpeg":
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)
    elif params.image_format=="png":
        image = tf.image.decode_png(parsed_features["image/encoded"], channels=3)
    else:
        raise ValueError("image_format: {}".format(params.image_format))

    if params.resize == True:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [params.image_size, params.image_size], align_corners=False)
        image = tf.squeeze(image, [0])
    else:
        image.set_shape([params.image_size, params.image_size, 3])

    image_normalized = tf.image.per_image_standardization(image)

    label = parsed_features["image/class/label"]

    if include_raw_image == True:
        return image, image_normalized, label
    else:
        return image_normalized, label


def input_fn(is_training, data_filenames, params):
    parse_fn = lambda example_proto: _parse_example(example_proto, params, False)

    if (is_training):
        dataset = (tf.data.TFRecordDataset(data_filenames)
            .map(parse_fn)
            .shuffle(buffer_size=params.shuffle_buffer_size)
            .repeat(params.num_epochs)
            .batch(params.batch_size)
            .prefetch(1))
    else:
        dataset = (tf.data.TFRecordDataset(data_filenames)
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1))

    return dataset

def raw_dataset(include_raw_image, data_filenames, params):
    parse_fn = lambda example_proto: _parse_example(example_proto, params, include_raw_image)

    dataset = (tf.data.TFRecordDataset(data_filenames)
        .map(parse_fn)
        .batch(params.batch_size)
        .prefetch(1))

    return dataset

def get_train_filenames(dataset_dir, dataset_name):
    train_data_filenames = []
    train_data_filenames.append(os.path.join(dataset_dir, 'train.tfrecord'))
    return train_data_filenames

def get_validation_filenames(dataset_dir, dataset_name):
    validation_data_filenames = []
    validation_data_filenames.append(os.path.join(dataset_dir, 'validation.tfrecord'))
    return validation_data_filenames
