import tensorflow as tf
import os
import dataset_zeus
import toolbox

dir_parent = os.path.dirname(os.getcwd())
name_food_gcld = '{}/food/gcld.tfrecords'.format(dir_parent)
name_food_gpld = '{}/food/gpld.tfrecords'.format(dir_parent)
name_food_gpld_vd = '{}/food/vd_gpld.tfrecords'.format(dir_parent)
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

x = tf.keras.layers.Lambda(lambda y: y[:, :, :, 0:256])(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(256, activation='elu',
                          kernel_initializer='he_normal',
                          bias_initializer='he_normal')(x)

x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(dataset_zeus.CONDITION_SUM, activation='linear')(x)

model_output = reshape(x)
head_model = tf.keras.Model(model_input, model_output)
head_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss='logcosh',
                   metrics=[
                       # tf.keras.metrics.mean_absolute_percentage_error,
                       tf.keras.metrics.mean_absolute_error,
                       dataset_zeus.my_accuracy])

head_model.summary()

callbacks = [
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir=dir_parent + '/logs/' + toolbox.get_now_str(),
                                   update_freq='epoch',
                                   write_graph=False,
                                   write_grads=True,
                                   histogram_freq=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=dir_parent + '/logs/temp_weights.h5', save_weights_only=True),
    # tf.keras.callbacks.LearningRateScheduler(schedule=toolbox.step_decay)
]

# train your model on data
test_set = dataset_zeus.create_dataset(name_food_gpld, shuffle_buffer=550)
vail_set = dataset_zeus.create_dataset(name_food_gpld_vd, shuffle_buffer=60)
head_model.fit(test_set, epochs=4, verbose=1, steps_per_epoch=100, callbacks=callbacks,
               batch_size=None, validation_data=vail_set, validation_steps=60)
# head_model.save(name_model)
