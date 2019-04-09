import csv
# import tensorflow as tf
import glob
import numpy as np


def read_csv(file_name, delimiter=','):
    products = csv.reader(open(file_name, newline=''), delimiter=delimiter, quotechar='|')
    my_list = []
    for row in products:
        dozen = []
        for item in row:
            if item is not '':
                dozen.append(float(item))
        my_list.append(dozen)
    return my_list


def write_csv(my_list, file_name, delimiter=','):
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=delimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item in my_list:
            csv_writer.writerow(item)


def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize_images(image, [500, 500])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_label(path):
    return read_csv(path)


def get_all_images_path(path):
    return sorted(glob.glob(path + '/*.png'))


def get_all_labels_path(path):
    return sorted(glob.glob(path + '/*label.csv'))


def get_all_cdts_path(path):
    return sorted(glob.glob(path + '/*condition.csv'))


def cook_images(paths):
    images = []
    for path in paths:
        images.append(load_and_preprocess_image(path))
    return np.asarray(images).reshape((len(images), 1))


def cook_labels(paths):
    labels = []
    for path in paths:
        labels.append(np.asarray(load_and_preprocess_label(path)))
    return np.asarray(labels).reshape((len(labels), 1))


def map_dataset(image_path, label):
    return load_and_preprocess_image(image_path), label
