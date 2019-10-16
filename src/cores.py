import tensorflow as tf
from darknet import DarkNet19, DarkNet53
from darknet import DWBottleNeck, DWBottleNeck2, ConvHeadEnd, ConnHeadEnd


def DWDark53(i_shape, b, a, weights=False):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet53(inputs))

    x = DWBottleNeck(darknet.output, filters=512)
    x = ConnHeadEnd(x, filters=b * a)
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWDark19(i_shape, b, a, weights=False):
    inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
    darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet19(inputs))

    x = DWBottleNeck(darknet.output, filters=512)
    x = ConnHeadEnd(x, filters=b * a)
    return tf.keras.Model(inputs=inputs, outputs=x)


def DWRes50(i_shape, b, a, weights=False):
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    if weights:
        tf.logging.warning('Loading ResNet50 Weights')
        resnet.load_weights('./weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = DWBottleNeck(resnet.output, filters=512)
    x = ConnHeadEnd(x, filters=b * a)
    core = tf.keras.Model(inputs=resnet.input, outputs=x)
    return core


def DWVGG19(i_shape, b, a, weights=False):
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=i_shape,
        weights=None)
    if weights:
        tf.logging.warning('Loading VGG19 Weights')
        vgg.load_weights('./weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = DWBottleNeck2(vgg.layers[-2].output, filters=512)
    x = ConnHeadEnd(x, filters=b * a)
    return tf.keras.Model(inputs=vgg.input, outputs=x)
