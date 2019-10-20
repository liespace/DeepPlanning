import tensorflow as tf
from darknet import DarkNet19, DarkNet53
from darknet import Bottleneck, Bottleneck2, HeadEnd


class Core:
    def __init__(self, config):
        # settings
        self.backbone_name = config['Model']['backbone']
        self.i_shape = tuple(config['Model']['i_shape'])
        self.a = config['Model']['A']
        self.b = config['Model']['B']
        self.weights = config['Train']['weights']
        # model
        self.core, self.backbone = self.buildup()

    def buildup(self):
        if self.backbone_name == 'dark53':
            print ('Running DWDark53')
            return self.DWDark53(
                i_shape=self.i_shape, a=self.a, b=self.b, weights=self.weights)
        elif self.backbone_name == 'res50':
            print ('Running DWRes50')
            return self.DWRes50(
                i_shape=self.i_shape, a=self.a, b=self.b, weights=self.weights)
        elif self.backbone_name == 'vgg19':
            tf.logging.warning('Running DWVGG19')
            return self.DWVGG19(
                i_shape=self.i_shape, a=self.a, b=self.b, weights=self.weights)
        elif self.backbone_name == 'dark19':
            print ('Running DWDark19')
            return self.DWDark19(
                i_shape=self.i_shape, a=self.a, b=self.b, weights=False)
        else:
            print ('Backbone name is wrong, Please re-checking')
            return None, None

    @staticmethod
    def DWDark53(i_shape, b, a, weights):
        inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
        darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet53(inputs))
        if weights:
            tf.logging.warning('Loading Dark53 Weights')
            darknet.load_weights('./weights/darknet53_weights.h5')

        x = Bottleneck(darknet.output, filters=1024)
        x = HeadEnd(x, filters=b * a)
        return tf.keras.Model(inputs=inputs, outputs=x), darknet

    @staticmethod
    def DWDark19(i_shape, b, a, weights=False):
        inputs = tf.keras.layers.Input(shape=i_shape, dtype=tf.float32)
        darknet = tf.keras.Model(inputs=inputs, outputs=DarkNet19(inputs))

        x = Bottleneck(darknet.output, filters=1024)
        x = HeadEnd(x, filters=b * a)
        return tf.keras.Model(inputs=inputs, outputs=x), darknet

    @staticmethod
    def DWRes50(i_shape, b, a, weights):
        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=i_shape,
            weights=None)
        if weights:
            tf.logging.warning('Loading ResNet50 Weights')
            resnet.load_weights('./weights/resnet50_weights.h5')

        x = Bottleneck(resnet.output, filters=10241725)
        x = HeadEnd(x, filters=b * a)
        core = tf.keras.Model(inputs=resnet.input, outputs=x), resnet
        return core

    @staticmethod
    def DWVGG19(i_shape, b, a, weights):
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            input_shape=i_shape,
            weights=None)
        if weights:
            tf.logging.warning('Loading VGG19 Weights')
            vgg.load_weights('./weights/vgg19_weights.h5')

        x = Bottleneck2(vgg.layers[-2].output, filters=1024)
        x = HeadEnd(x, filters=b * a)
        return tf.keras.Model(inputs=vgg.input, outputs=x), vgg
