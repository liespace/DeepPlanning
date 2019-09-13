import tensorflow as tf
from functools import wraps, reduce
import os
import json

cfg_path = os.getcwd() + os.sep + 'src' + os.sep + 'config.json'
with open(cfg_path) as handle:
    config = json.loads(handle.read())
LAM = config['Loss']['lamb']


def Compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(tf.keras.layers.Conv2D)
def DarkConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(LAM),
                           'padding': 'valid' if kwargs.get('strides') == (
                               2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)


@wraps(tf.keras.layers.Conv2D)
def DWConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(LAM),
                           'padding': 'valid'}
    darknet_conv_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)


def DarkConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return Compose(DarkConv2D(*args, **no_bias_kwargs),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(alpha=0.1))


def DWConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return Compose(DWConv2D(*args, **no_bias_kwargs),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(alpha=0.1))


def ResBlock(x, num_filters, num_blocks):
    """A series of res-blocks starting with a down-sampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarkConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = Compose(DarkConv2D_BN_Leaky(num_filters // 2, (1, 1)),
                    DarkConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = tf.keras.layers.Add()([x, y])
    return x


def BottleNeck(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    return Compose(DarkConv2D_BN_Leaky(outer_filters, (3, 3)),
                   DarkConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
                   DarkConv2D_BN_Leaky(outer_filters, (3, 3)))


def BottleNeck2(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return Compose(BottleNeck(outer_filters, bottleneck_filters),
                   DarkConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
                   DarkConv2D_BN_Leaky(outer_filters, (3, 3)))


def DarkNet19(x):
    """Generate first 18 conv layers of Darknet-19."""
    backbone = Compose(
        DarkConv2D_BN_Leaky(32, (3, 3)),
        tf.keras.layers.MaxPooling2D(),
        DarkConv2D_BN_Leaky(64, (3, 3)),
        tf.keras.layers.MaxPooling2D(),
        BottleNeck(128, 64),
        tf.keras.layers.MaxPooling2D(),
        BottleNeck(256, 128),
        tf.keras.layers.MaxPooling2D(),
        BottleNeck2(512, 256),
        tf.keras.layers.MaxPooling2D(),
        BottleNeck2(1024, 512))
    return backbone(x)


def DarkNet53(x):
    """DarkNet body having 52 Convolution2D layers"""
    x = DarkConv2D_BN_Leaky(32, (3, 3))(x)
    x = ResBlock(x, 64, 1)
    x = ResBlock(x, 128, 2)
    x = ResBlock(x, 256, 8)
    x = ResBlock(x, 512, 8)
    x = ResBlock(x, 1024, 4)
    return x


def Bottleneck(x, filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = Compose(DarkConv2D_BN_Leaky(filters / 4, (1, 1)),
                DarkConv2D_BN_Leaky(filters / 2, (3, 3)),
                DarkConv2D_BN_Leaky(filters / 8, (1, 1)),
                DarkConv2D_BN_Leaky(filters / 4, (3, 3)),
                DarkConv2D_BN_Leaky(filters / 16, (1, 1)))(x)
    return x


def Bottleneck2(x, filters):
    x = Compose(
        tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0))),
        DarkConv2D_BN_Leaky(filters, (3, 3), strides=(2, 2)),
        DarkConv2D_BN_Leaky(filters / 4, (1, 1)),
        DarkConv2D_BN_Leaky(filters / 2, (3, 3)),
        DarkConv2D_BN_Leaky(filters / 8, (1, 1)),
        DarkConv2D_BN_Leaky(filters / 4, (3, 3)),
        DarkConv2D_BN_Leaky(filters / 16, (1, 1)))(x)
    return x


def HeadEnd(x, filters):
    return Compose(
        DarkConv2D(filters, (1, 1)),
        DWConv2D_BN_Leaky(filters, (3, 3)),
        DWConv2D_BN_Leaky(filters, (3, 3)),
        DWConv2D_BN_Leaky(filters, (3, 3)),
        DWConv2D_BN_Leaky(filters, (3, 3)),
        DWConv2D_BN_Leaky(filters, (3, 3)),
        DWConv2D_BN_Leaky(filters, (3, 3)),
        tf.keras.layers.Conv2D(filters, (3, 3), activation='sigmoid'))(x)


def Concat(x0, x1, x0_filters):
    x0 = Compose(DarkConv2D_BN_Leaky(x0_filters, (1, 1)),
                 tf.keras.layers.UpSampling2D(2))(x0)
    return tf.keras.layers.Concatenate()([x0, x1])


def Concat2(x0, x1, x0_filters, x1_filters):
    x0 = Compose(
        DarkConv2D_BN_Leaky(x0_filters, (1, 1)),
        tf.keras.layers.UpSampling2D(2))(x0)
    x1 = DarkConv2D_BN_Leaky(x1_filters, (1, 1))(x1)
    return tf.keras.layers.Concatenate()([x0, x1])
