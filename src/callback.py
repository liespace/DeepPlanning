import tensorflow as tf
from monitor import DataMonitor


class PredictionCheckPoint(tf.keras.callbacks.Callback):
    def __init__(self, dir_parent=None):
        super().__init__()
        self.dir_parent = dir_parent
        self.monitor = DataMonitor(dir_parent=self.dir_parent + '/dataset',
                                   menu={'gridmap', 'condition', 'label', 'prediction'})

    def on_epoch_end(self, epoch, logs=None):
        print('hello {}'.format(epoch))
        these = [0, 50, 140, 220, 300, 390, 500, 610]
        for this in these:
            number = this
            data = self.monitor.read_input(number)
            prediction = self.model.predict(data, batch_size=1, steps=1)
            self.monitor.write_prediction(number=number, data=prediction[0, :, :] + data[1][0, :, :])
        self.monitor.show(mode='one', which=these, layout=441, name=epoch, gui=False)
