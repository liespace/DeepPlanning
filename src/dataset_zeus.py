import tensorflow as tf
import numpy as np

dict_gcld = {'gridmap': tf.FixedLenFeature(shape=(), dtype=tf.string),
             'condition': tf.FixedLenFeature(shape=(), dtype=tf.string),
             'label': tf.FixedLenFeature(shape=(), dtype=tf.string),
             'delta': tf.FixedLenFeature(shape=(), dtype=tf.string)}

dict_gpld = {'gridmap': tf.FixedLenFeature(shape=(), dtype=tf.string),
             'condition': tf.FixedLenFeature(shape=(), dtype=tf.string),
             'label': tf.FixedLenFeature(shape=(), dtype=tf.string),
             'delta': tf.FixedLenFeature(shape=(), dtype=tf.string)}

DICT_DATASET = dict_gpld
CONDITION_SHAPE = (4, 3)
CONDITION_SUM = CONDITION_SHAPE[0] * CONDITION_SHAPE[1]
GRIDMAP_SHAPE = (500, 500, 3)
SHUFFLE_BUFFER = 610
BATCH_SIZE = 1


def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=4)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    gridmap, delta = iterator.get_next()

    return dataset


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = dict_gcld

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    parsed_features['gridmap'] = tf.decode_raw(parsed_features['gridmap'], tf.float32)
    parsed_features['gridmap'] = tf.reshape(parsed_features['gridmap'], GRIDMAP_SHAPE) / 255

    parsed_features['delta'] = tf.decode_raw(parsed_features['delta'], tf.float32)
    parsed_features['delta'] = tf.reshape(parsed_features['delta'], CONDITION_SHAPE)

    return parsed_features['gridmap'], parsed_features['delta']


def my_accuracy(y_true, y_pred, **kwargs):
    threshold = [0.5, 0.5, np.radians(3)]
    limit = CONDITION_SUM

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    threshold = tf.cast(threshold, y_pred.dtype)
    limit = tf.cast(limit, y_pred.dtype)

    x = tf.abs(y_pred - y_true)
    x = tf.cast(x < threshold, y_pred.dtype)
    x = tf.reduce_sum(x, axis=-1)
    x = tf.reduce_sum(x, axis=-1)
    x = tf.cast(x >= limit, tf.float32)
    x = tf.keras.backend.mean(x, axis=-1)
    return x
