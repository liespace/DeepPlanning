import tensorflow as tf


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('hello {}'.format(epoch))
