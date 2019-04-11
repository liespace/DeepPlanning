import tensorflow as tf
import os
import dataset_zeus
import toolbox

dir_parent = os.path.dirname(os.getcwd())
name_food_gcld = '{}/food/gcld.tfrecords'.format(dir_parent)
name_food_gpld = '{}/food/gpld.tfrecords'.format(dir_parent)
name_model = '{}/food/model_zeus.h5'.format(dir_parent)

# cook model
image_processor = tf.keras.applications.ResNet50(weights=None, include_top=False,
                                                 input_shape=dataset_zeus.GRIDMAP_SHAPE)
image_processor.load_weights(dir_parent + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

reshape = tf.keras.Sequential()
reshape.add(tf.keras.layers.Reshape(dataset_zeus.CONDITION_SHAPE,
                                    input_shape=(dataset_zeus.CONDITION_SUM,)))

model_input = tf.keras.layers.Input(shape=dataset_zeus.GRIDMAP_SHAPE, dtype=tf.float32, name='main_input')
x = image_processor(model_input)

x = tf.keras.layers.DepthwiseConv2D(kernel_size=16, strides=(1, 1),
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    bias_initializer='he_normal')(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(150, activation='relu',
                          kernel_initializer='he_normal',
                          bias_initializer='he_normal')(x)

x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(dataset_zeus.CONDITION_SUM, activation='linear',
                          kernel_initializer='glorot_normal',
                          bias_initializer='glorot_normal')(x)

model_output = reshape(x)

head_model = tf.keras.Model(model_input, model_output)
head_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss='mean_squared_error',
                   metrics=[tf.keras.metrics.mean_absolute_error,
                            tf.keras.metrics.mean_absolute_percentage_error,
                            dataset_zeus.my_accuracy])

head_model.summary()

callbacks = [
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir=dir_parent + '/logs/' + toolbox.get_now_str(), update_freq='epoch'),
    tf.keras.callbacks.ModelCheckpoint(filepath=dir_parent + '/temp_weights.h5', save_weights_only=True)
]

# train your model on data
dataset = dataset_zeus.create_dataset(name_food_gpld)
head_model.fit(dataset, epochs=100, verbose=1, steps_per_epoch=10, callbacks=callbacks)
# head_model.save(name_model)
