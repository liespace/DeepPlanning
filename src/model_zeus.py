import tensorflow as tf
import toolbox
import os
import random

dics = {'data': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'label': tf.FixedLenFeature(shape=(), dtype=tf.string)}


def parse_example(example):
    parsed_example = tf.parse_single_example(example, dics)
    parsed_example['data'] = tf.decode_raw(parsed_example['data'], tf.float32)
    parsed_example['data'] = tf.reshape(parsed_example['data'], (500, 500, 3))
    parsed_example['label'] = tf.decode_raw(parsed_example['label'], tf.float32)
    parsed_example['label'] = tf.reshape(parsed_example['label'], (5, 3))
    return parsed_example


dir_parent = os.path.dirname(os.getcwd())
name_food_cd = '{}/food/cd.tfrecords'.format(dir_parent)
name_food_pd = '{}/food/pd.tfrecords'.format(dir_parent)

food = tf.data.TFRecordDataset(name_food_cd)
new_dataset = food.map(parse_example)
print(new_dataset.output_types)
print(new_dataset.output_shapes)
print(new_dataset.output_classes)
# iterator = new_dataset.make_one_shot_iterator()
# sess = tf.InteractiveSession()
# next_element = iterator.get_next()
# i = 0
# while True:
#     try:
#         data, label = sess.run([next_element['data'],
#                                 next_element['label']])
#     except tf.errors.OutOfRangeError:
#         print("End of dataset")
#         break
#     else:
#         print(i)
#         print(data.shape)
#         print(label.shape)
#     i += 1

# ds = food.cache(filename='./cache.tf-data')
# ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(600)))
# ds = ds.batch(32).prefetch(1)

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
