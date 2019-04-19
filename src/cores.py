import tensorflow as tf
import numpy as np

"""
D:
> In[0]
    |-Resnet50
        |-Filter(:, :, 0:256)
                |-Flatten
                    |
> In[1]             |
    |-Dense(128)    |
        |-----------|
> In[2]             |
    |-Dense(128)    |
        |-----------|-Merge
                        |-Dense(256)
                            |-Dropout(0.5)
                                |-Reshape
                                    |-Out >>>>>>>>
G:
> In[0]
    |-Resnet50
        |-Filter(:, :, 0:256)
                |-Flatten
                    |
> In[1]             |
    |-Dense(128)    |
        |-----------|
> In[2]             |
    |---------------|
                    |-Merge
                        |-Dense(256)
                            |-Dropout(0.5)
                                |-Reshape
                                    |-Out  >>>>>>>
"""


def gcore(input_shape, output_shape, ipu, oru):
    return None


def dcore(input_shape, output_shape, ipu, oru):
    return None


def beta(input_shape, output_shape, ipu, oru):
    """
    > In_Main
        |-Resnet50()
            |-Filter(:, :, 0:256)
                |-Flatten
                    |-BN
                        |-Dense(256)
    > In_Aux                |
        |-Flatten           |
            |-Dense(128)    |
                |-----------|-Merge
                                |-BN
                                    |-Dense(256)
                                        |-Dropout(0.5)
                                            |-Reshape
                                                |-Out >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    in_main = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='in_main')
    x0 = ipu(in_main)
    x0 = tf.keras.layers.Lambda(lambda y: y[:, :, :, 0:256])(x0)
    x0 = tf.keras.layers.Flatten()(x0)
    x0 = tf.keras.layers.BatchNormalization()(x0)
    x0 = tf.keras.layers.Dense(256, activation='elu', kernel_initializer='he_normal', bias_initializer='he_normal')(x0)

    in_aux = tf.keras.layers.Input(shape=output_shape, dtype=tf.float32, name='in_aux')
    x1 = tf.keras.layers.Flatten()(in_aux)
    x1 = tf.keras.layers.Dense(128, activation='elu', kernel_initializer='he_normal', bias_initializer='he_normal')(x1)

    x = tf.keras.layers.concatenate([x0, x1])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='elu', kernel_initializer='he_normal', bias_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(np.prod(output_shape), activation='linear')(x)
    out = oru(x)

    core = tf.keras.Model(inputs=[in_main, in_aux], outputs=out)
    return core


def alpha(input_shape, output_shape, ipu, oru):
    """
    > In
        |-Resnet50()
            |-Filter(:, :, 0:256)
                |-Flatten
                    |-BN
                        |-Dense(256)
                            |-Dropout(0.5)
                                |-Reshape
                                    |-Out >>>>>>>>>>>>>>>>>>>>>>
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
