import tensorflow as tf
from darknet import DarkNet19, DarkNet53
from darknet import Bottleneck, Bottleneck2, HeadEnd


def DWDark53(i_shape, b, c, a):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet53(inputs))

    x = Bottleneck(darknet.output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWDark19(i_shape, b, c, a):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet19(inputs))

    x = Bottleneck(darknet.output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWRes50(i_shape, b, c, a):
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    x = Bottleneck(resnet.output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    return tf.keras.Model(inputs=resnet.input, outputs=x)


def DWVGG19(i_shape, b, c, a):
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    x = Bottleneck2(vgg.layers[-2].output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    return tf.keras.Model(inputs=vgg.input, outputs=x)
