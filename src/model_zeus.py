import tensorflow as tf
import toolbox
import os
import random

parent_folder = os.path.dirname(os.getcwd())
data_path = '{}/dataset'.format(parent_folder)
image_paths = toolbox.get_all_images_path(data_path)
label_paths = toolbox.get_all_labels_path(data_path)
cdts_paths = toolbox.get_all_cdts_path(data_path)

labels = tf.data.experimental.CsvDataset(label_paths, [tf.float32] * 3)
cdts = tf.data.experimental.CsvDataset(cdts_paths, [tf.float32] * 3, header=True)

labels = labels - cdts
print(labels)

images = tf.data.Dataset.from_tensor_slices(image_paths)

ds = tf.data.Dataset.zip((images, labels))
image_label_ds = ds.map(toolbox.map_dataset)
print(image_label_ds)

ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(image_paths)))
ds = ds.batch(32).prefetch(1)

# network overall params
shape_input = (500, 500, 3)
dim_output = 3
num_output = 4

# training related params
batch_size = 1000
epochs = 10
verbose = 1

# cook dataset
data = []
labels = []

# # cook model
# base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=shape_input)
# base_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
# layer = tf.keras.layers.Flatten()(base_model.output)
# layer = tf.keras.layers.Dense(4096, activation='relu')(layer)
# layer = tf.keras.layers.Dropout(0.5)(layer)
# layer = tf.keras.layers.BatchNormalization()(layer)
# prediction = tf.keras.layers.Dense(num_output * dim_output, activation='relu')(layer)
#
# head_model = tf.keras.Model(base_model.input, prediction)
# head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# head_model.summary()

# train your model on data
# head_model.fit(x=data, y=labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
