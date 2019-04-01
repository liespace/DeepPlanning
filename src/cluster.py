import os
import numpy as np
import tensorflow as tf
import toolbox as tl

no = 34
parent_folder = os.path.dirname(os.getcwd())
print(parent_folder)
products_file_name = '{}/dataset/{}products.csv'.format(parent_folder, no)
condition_file_name = '{}/dataset/{}condition.csv'.format(parent_folder, no)
points = np.asarray(tl.read_csv(products_file_name))
previous_centers = np.asarray(tl.read_csv(condition_file_name, ' '))[1:, :]
num_clusters = 4
num_iterations = 10


def input_fn():
    return tf.data.Dataset.from_tensors(
        tf.convert_to_tensor(points, dtype=tf.float32)).repeat(1)


kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)

# train
for _ in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
        print('delta: {}'.format(cluster_centers - previous_centers))
    previous_centers = cluster_centers
    print('score: {}'.format(kmeans.score(input_fn)))
print('cluster centers: {}'.format(cluster_centers))

# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    print('point: {}, is in cluster {}, centered at{}'.format(point, cluster_index, center))
