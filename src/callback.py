import tensorflow as tf


class WarmUpLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, warm_up, schedule):
        super(WarmUpLRSchedule, self).__init__()
        self.epoch = 0
        self.warm_up = warm_up
        self.schedule = schedule

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.epoch < 1:
            # Get the current learning rate from model's optimizer.
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            # Call schedule function to get the scheduled learning rate.
            lrd = self.warm_up(batch)
            # Set the value back to the optimizer before this epoch starts
            tf.keras.backend.set_value(self.model.optimizer.lr, lrd)
            # print('\nBatch %05d: LR is %6.4f from %6.4f.' % (batch, lrd, lr))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        lrd = self.schedule(epoch)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, lrd)
        print('\nEpoch %05d: LR is %6.4f from %6.4f.' % (epoch, lrd, lr))
