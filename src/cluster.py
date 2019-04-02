import os
import numpy as np
import tensorflow as tf
import toolbox as tl


no = 0
while no <= 633:
    parent_folder = os.path.dirname(os.getcwd())
    products_file_name = '{}/dataset/{}products.csv'.format(parent_folder, no)
    condition_file_name = '{}/dataset/{}condition.csv'.format(parent_folder, no)
    label_file_name = '{}/dataset/{}label.csv'.format(parent_folder, no)
    points = np.asarray(tl.read_csv(products_file_name))
    previous_centers = np.asarray(tl.read_csv(condition_file_name, ' '), np.float32)
    num_clusters = 5
    num_iterations = 10

    if points.size == 0:
        no += 1
        continue


    def input_fn():
        return tf.data.Dataset.from_tensors(tf.convert_to_tensor(points, dtype=tf.float32)).repeat(1)


    kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters,
                                                       initial_clusters=previous_centers,
                                                       use_mini_batch=False)

    # train
    for _ in range(num_iterations):
        kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()
        delta = cluster_centers - previous_centers
        previous_centers = cluster_centers
        if (delta < 1e-5).all() and (delta > -1e-5).all():
            print('delta: {}'.format(delta))
            print('score: {}'.format(kmeans.score(input_fn)))
            break
    print('cluster centers: {}'.format(cluster_centers))

    # save cluster centers
    tl.write_csv(list(cluster_centers), label_file_name)
    print('cluster centers for SET {} saved'.format(no))

    # # map the input points to their clusters
    # cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    # for i, point in enumerate(points):
    #     cluster_index = cluster_indices[i]
    #     center = cluster_centers[cluster_index]
    #     print('point: {}, is in cluster {}, centered at{}'.format(point, cluster_index, center))

    no += 1
