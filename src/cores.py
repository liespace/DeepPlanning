import tensorflow as tf
from darknet import DarkNet19, DarkNet53
from darknet import FrontEnd, Concat, Concat2, HeadEnd2, HeadEnd3


def DWDark53(i_shape, b, c, a):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet53(inputs))

    x = FrontEnd(darknet.output, filters=512)
    x = Concat(x, darknet.layers[152].output, x0_filters=256)
    x = HeadEnd2(x, filters=256, o_filters=b * (c + a))
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWDark19(i_shape, b, c, a):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet19(inputs))

    x = FrontEnd(darknet.output, filters=512)
    x = Concat(x, darknet.layers[-17].output, x0_filters=256)
    x = HeadEnd2(x, filters=256, o_filters=b * (c + a))
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWRes50(i_shape, b, c, a):
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    x = FrontEnd(resnet.output, filters=512)
    x = Concat2(x, resnet.layers[-33].output, x0_filters=256, x1_filters=512)
    x = HeadEnd2(x, filters=256, o_filters=b * (c + a))
    core = tf.keras.Model(inputs=resnet.input, outputs=x)
    return core


def DWVGG19(i_shape, b, c, a):
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    x = FrontEnd(vgg.layers[-2].output, filters=256)
    x = HeadEnd3(x, filters=128, o_filters=b * (c + a))
    core = tf.keras.Model(inputs=vgg.input, outputs=x)
    return core
