import tensorflow as tf
import numpy as np


def alpha(input_shape, output_shape, ipu, oru):
    """
    > Input
        |-Resnet50()
            |-Filter(:, :, 0:256)
                |-Flatten
                    |-BN
                        |-Dense(256)
                            |-Dropout(0.5)
                                |-Output >>
    """
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='input')
    x = ipu(inputs)
    x = tf.keras.layers.Lambda(lambda y: y[:, :, :, 0:256])(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='elu', kernel_initializer='he_normal', bias_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(np.prod(output_shape), activation='linear')(x)
    outputs = oru(x)
    core = tf.keras.Model(inputs, outputs)
    return core
