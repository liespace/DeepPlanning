import tensorflow as tf
import numpy as np
from darknet import DarkNet19, DarkNet53, FrontEnd, DarkConv2D_BN_Leaky, Compose


def DWYolo(i_shape, b, c, a):
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet53(inputs))

    x, y1 = FrontEnd(darknet.output, 512, b * (c + a))

    x = Compose(DarkConv2D_BN_Leaky(256, (1, 1)),
                tf.keras.layers.UpSampling2D(2))(y1)
    x = tf.keras.layers.Concatenate()([x, darknet.layers[152].output])
    x, y2 = FrontEnd(x, 256, b * (c + a))

    return tf.keras.Model(inputs=inputs, outputs=y2)


def DWRes50(i_shape):
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
    fen = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    backbone = tf.keras.Model(inputs=fen.input, outputs=fen.layers[-3].output)
    x = fen(inputs)
    core = tf.keras.Model(inputs=backbone.input, outputs=backbone.output)
    return core


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
    in_main = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32,
                                    name='in_main')
    x0 = ipu(in_main)
    x0 = tf.keras.layers.Lambda(lambda y: y[:, :, :, 0:256])(x0)
    x0 = tf.keras.layers.Flatten()(x0)
    x0 = tf.keras.layers.BatchNormalization()(x0)
    x0 = tf.keras.layers.Dense(256, activation='elu',
                               kernel_initializer='he_normal',
                               bias_initializer='he_normal')(x0)

    in_aux = tf.keras.layers.Input(shape=output_shape, dtype=tf.float32,
                                   name='in_aux')
    x1 = tf.keras.layers.Flatten()(in_aux)
    x1 = tf.keras.layers.Dense(128, activation='elu',
                               kernel_initializer='he_normal',
                               bias_initializer='he_normal')(x1)

    x = tf.keras.layers.concatenate([x0, x1])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='elu',
                              kernel_initializer='he_normal',
                              bias_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(np.prod(output_shape), activation='linear')(x)
    out = oru(x)

    core = tf.keras.Model(inputs=[in_main, in_aux], outputs=out)
    return core
