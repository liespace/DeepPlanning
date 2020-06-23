import os
import logging
import tensorflow as tf
from .subnet import DarkNet19, DarkNet53, Config, SVG16
from .subnet import (Bottleneck, Bottleneck2, HeadEnd,
                     DWBottleNeck, ConnHeadEnd, DWBottleNeck2)


class Core:
    def __init__(self, config):
        Config.LAM = config['Loss']['lamb']
        self.a = config['Model']['A']
        self.b = config['Model']['B']
        self.tiny = config['Model']['tiny']
        self.backbone_name = config['Model']['backbone']
        self.i_shape = tuple(config['Model']['i_shape'])
        self.load_pretrained_weights = config['Train']['load_pretrained_weights']
        self.pretrained_weights_folder = config['Train']['pretrained_weights_folder']
        self.core, self.backbone = self.buildup()

    def buildup(self):
        if self.backbone_name == 'dark53':
            logging.info('Running with Backbone DarkNet-53')
            return self.DWDark53(
                tiny=self.tiny,
                i_shape=self.i_shape, a=self.a, b=self.b,
                load_pretrained_weights=self.load_pretrained_weights)
        elif self.backbone_name == 'res50':
            logging.info('Running with Backbone ResNet-50')
            return self.DWRes50(
                tiny=self.tiny,
                i_shape=self.i_shape, a=self.a, b=self.b,
                load_pretrained_weights=self.load_pretrained_weights)
        elif self.backbone_name == 'vgg19':
            logging.info('Running with Backbone VGG-19')
            return self.DWVGG19(
                tiny=self.tiny,
                i_shape=self.i_shape, a=self.a, b=self.b,
                load_pretrained_weights=self.load_pretrained_weights)
        elif self.backbone_name == 'vgg16':
            logging.info('Running with Backbone VGG-16')
            return self.DWVGG16(
                tiny=self.tiny,
                i_shape=self.i_shape, a=self.a, b=self.b,
                load_pretrained_weights=self.load_pretrained_weights)
        elif self.backbone_name == 'xception':
            logging.info('Running with Backbone Xception')
            return self.Xception(
                tiny=self.tiny,
                i_shape=self.i_shape, a=self.a, b=self.b,
                load_pretrained_weights=self.load_pretrained_weights)
        elif self.backbone_name == 'svg16':
            logging.info('Running with Backbone SVG16 (VGG-16 of SSD object detector)')
            return self.DWSVG16(
                tiny=self.tiny,
                i_shape=self.i_shape, a=self.a, b=self.b,
                load_pretrained_weights=self.load_pretrained_weights)
        else:
            logging.warning('Backbone name is wrong, Please re-checking')
            return None, None

    def DWRes50(self, i_shape, b, a, load_pretrained_weights, tiny):
        resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=i_shape, weights=None)
        if load_pretrained_weights:
            logging.warning('Loading ResNet50 Weights')
            resnet.load_weights(self.pretrained_weights_folder + os.sep + 'resnet50_weights.h5')
        if tiny:
            x = Bottleneck(resnet.output, filters=1024)
            x = HeadEnd(x, filters=b * a)
        else:
            x = DWBottleNeck(resnet.output, filters=512)
            x = ConnHeadEnd(x, filters=b * a)
        return (tf.keras.Model(inputs=resnet.input, outputs=x),
                tf.keras.Model(inputs=resnet.input, outputs=resnet.get_layer('add_14').output))

    def DWVGG19(self, i_shape, b, a, load_pretrained_weights, tiny):
        vgg = tf.keras.applications.VGG19(include_top=False, input_shape=i_shape, weights=None)
        if load_pretrained_weights:
            logging.warning('Loading VGG19 Weights')
            vgg.load_weights(self.pretrained_weights_folder + os.sep + 'vgg19_weights.h5')
        if tiny:
            x = Bottleneck2(vgg.layers[-2].output, filters=1024)
            x = HeadEnd(x, filters=b * a)
        else:
            x = DWBottleNeck2(vgg.layers[-2].output, filters=512)
            x = ConnHeadEnd(x, filters=b * a)
        return tf.keras.Model(inputs=vgg.input, outputs=x), vgg

    def DWVGG16(self, i_shape, b, a, load_pretrained_weights, tiny):
        vgg = tf.keras.applications.VGG16(include_top=False, input_shape=i_shape, weights=None)
        if load_pretrained_weights:
            logging.warning('Loading VGG16 Weights')
            vgg.load_weights(self.pretrained_weights_folder + os.sep + 'vgg16_weights.h5')
        if tiny:
            x = Bottleneck2(vgg.layers[-2].output, filters=1024)
            x = HeadEnd(x, filters=b * a)
        else:
            x = DWBottleNeck2(vgg.layers[-2].output, filters=512)
            x = ConnHeadEnd(x, filters=b * a)
        return tf.keras.Model(inputs=vgg.input, outputs=x), vgg

    def DWDark53(self, i_shape, b, a, load_pretrained_weights, tiny):
        inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
        darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet53(inputs))
        if load_pretrained_weights:
            logging.warning('Loading Dark53 Weights')
            darknet.load_weights(self.pretrained_weights_folder + os.sep + 'darknet53_weights.h5')
        if tiny:
            x = Bottleneck(darknet.output, filters=1024)
            x = HeadEnd(x, filters=b * a)
        else:
            x = DWBottleNeck(darknet.output, filters=512)
            x = ConnHeadEnd(x, filters=b * a)
        return (tf.keras.Model(inputs=inputs, outputs=x),
                tf.keras.Model(inputs=darknet.input, outputs=darknet.get_layer('add_21').output))

    def Xception(self, i_shape, b, a, tiny, load_pretrained_weights=False):
        xception = tf.keras.applications.Xception(include_top=False, input_shape=i_shape, weights=None)
        if load_pretrained_weights:
            logging.warning('Loading Xception Weights')
            xception.load_weights(self.pretrained_weights_folder + os.sep + 'xception_weights.h5')
            xception.summary()
        if tiny:
            x = Bottleneck(xception.output, filters=1024)
            x = HeadEnd(x, filters=b * a)
        else:
            x = DWBottleNeck(xception.output, filters=512)
            x = ConnHeadEnd(x, filters=b * a)
        return (tf.keras.Model(inputs=xception.input, outputs=x),
                tf.keras.Model(inputs=xception.input, outputs=xception.get_layer('add_11').output))

    def DWSVG16(self, i_shape, b, a, load_pretrained_weights, tiny):
        inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
        svg = tf.keras.Model(inputs=inputs, outputs=SVG16(inputs))
        if load_pretrained_weights:
            logging.warning('Loading SVG16 Weights')
            svg.load_weights(self.pretrained_weights_folder+os.sep+'VGG_VOC0712_SSD_512x512_iter_120000.h5', by_name=True)
        if tiny:
            x = Bottleneck2(svg.output, filters=1024)
            x = HeadEnd(x, filters=b * a)
        else:
            x = DWBottleNeck2(svg.output, filters=512)
            x = ConnHeadEnd(x, filters=b * a)
        return tf.keras.Model(inputs=svg.input, outputs=x), svg
