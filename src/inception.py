import tensorflow as tf
from monitor import DataMonitor
from dataset import DatasetHolder
import os

model_id = 'ml-rrt-1'
dir_parent = os.path.dirname(os.getcwd())
checkpoint = '{}/logs/{}/'.format(dir_parent, model_id) + 'checkpoint.h5'
model = tf.keras.models.load_model(checkpoint, custom_objects={'my_accuracy': DatasetHolder().my_accuracy})
model.summary()
monitor = DataMonitor(dir_parent=dir_parent + '/dataset',
                      menu={'gridmap', 'condition', 'prediction'})
these = [0, 50, 140, 220, 300, 390, 500, 610]
for this in these:
    number = this
    data = monitor.read_input(number)
    prediction = model.predict(data, batch_size=1, steps=1)
    monitor.write_prediction(number=number, data=prediction[0, :, :] + data[1][0, :, :])
monitor.show(mode='one', which=these, layout=441, name=45000, gui=True)
