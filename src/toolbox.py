import csv
import tensorflow as tf
import glob
import numpy as np
from datetime import datetime


def get_now_str():
    return str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')


def gcs_to_lcs(condition, label):
    state = condition[0, 0:2]
    angle = condition[0, 2]
    rotate = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])

    condition_location = condition[:, 0:2]
    condition_yaw = condition[:, 2]
    label_location = label[:, 0:2]
    label_yaw = label[:, 2]

    condition_location = np.dot((condition_location - state), rotate.transpose())
    condition_yaw = condition_yaw.transpose() - angle
    label_location = np.dot((label_location - state), rotate.transpose())
    label_yaw = label_yaw - angle

    lcs_condition = np.c_[condition_location, condition_yaw]
    lcs_label = np.c_[label_location, label_yaw]
    lcs_label[0, :] = lcs_condition[0, :]

    return lcs_condition, lcs_label


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


def show_dataset(dataset):
    print(dataset.output_types)
    print(dataset.output_shapes)
    print(dataset.output_classes)

    iterator = dataset.make_one_shot_iterator()
    sess = tf.InteractiveSession()
    next_element = iterator.get_next()
    i = 0
    while True:
        try:
            gridmap, condition, delta = sess.run([next_element[0],
                                                  next_element[1],
                                                  next_element[2]])
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
        else:
            print(i)
            print(gridmap.shape)
            print(condition.shape)
            print(delta.shape)
        if i is 0:
            print(condition)
            print(delta)
        i += 1
