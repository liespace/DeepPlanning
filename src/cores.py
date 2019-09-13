import tensorflow as tf
from darknet import DarkNet19, DarkNet53
from darknet import Bottleneck, Bottleneck2, HeadEnd


def DWDark53(i_shape, b, c, a, weights=False):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet53(inputs))

    x = Bottleneck(darknet.output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWDark19(i_shape, b, c, a, weights=False):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet19(inputs))

    x = Bottleneck(darknet.output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWRes50(i_shape, b, c, a, weights=False):
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    if weights:
        tf.logging.info('Loading ResNet50 Weights')
        resnet.load_weights('./weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = Bottleneck(resnet.output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    core = tf.keras.Model(inputs=resnet.input, outputs=x)
    return core


def DWVGG19(i_shape, b, c, a, weights=False):
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    if weights:
        tf.logging.warning('Loading VGG19 Weights')
        vgg.load_weights('./weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = Bottleneck2(vgg.layers[-2].output, filters=1024)
    x = HeadEnd(x, filters=b * (c + a))
    return tf.keras.Model(inputs=vgg.input, outputs=x)
