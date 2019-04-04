import tensorflow as tf

class ModelAlpha(tf.keras.Model):

    def __init__(self):
        super(ModelAlpha, self).__init__(name='model_alpha')
        # image layers
        self.imagenet = tf.keras.applications.ResNet50(include_top=True, weights=None)
        # fusion layers
        self.fusion_1 = tf.keras.layers.Dense(36, activation='relu')
        # end layers
        self.end_1 = tf.keras.layers.Dense(12, activation='softmax')

    def call(self, inputs, **kwargs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.imagenet(inputs)
        x = self.fusion_1(x)
        return self.end_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


class ModelBeta(tf.keras.Model):

    def __init__(self):
        super(ModelBeta, self).__init__(name='model_beta')
        # image layers
        self.imagenet = tf.keras.applications.ResNet50(include_top=True, weights=None)
        # fusion layers
        self.fusion_1 = tf.keras.layers.Dense(36, activation='relu')
        # end layers
        self.end_1 = tf.keras.layers.Dense(12, activation='softmax')

    def call(self, inputs, **kwargs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.imagenet(inputs)
        x = tf.keras.layers.concatenate([x, inputs])
        x = self.fusion_1(x)
        return self.end_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


print(tf.VERSION)
print(tf.keras.__version__)
