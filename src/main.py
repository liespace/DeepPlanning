from dataset import DatasetHolder
from model import Model
import tensorflow as tf
import cores
import numpy as np

train_size = 542
validation_size = 60
batch_size = 1
train_steps = np.ceil(train_size / batch_size)
validation_steps = np.ceil(validation_size / batch_size)

keeper = DatasetHolder(food_type='gpld', menu=['gridmap', 'delta'])
train_set = keeper.create_dataset(use_for='train', shuffle_buffer=550, batch_size=batch_size)
validation_set = keeper.create_dataset(use_for='validation', shuffle_buffer=60, batch_size=batch_size)
ipu_weight = keeper.dir_parent + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Model(build_core=cores.alpha, name='alpha',
              train_set=train_set, validation_set=validation_set,
              input_shape=keeper.gridmap_shape, output_shape=keeper.label_shape,
              ipu_weight=ipu_weight, verbose=1,

              optimizer=tf.keras.optimizers.Adam(),
              loss='logcosh',
              metrics=[tf.keras.metrics.mean_absolute_error, keeper.my_accuracy],
              checkpoint=True, check_period=100,
              tensorboard=True,

              epochs=6000,
              steps_per_epoch=train_steps,
              batch_size=None,
              validation_steps=validation_steps,

              lr_step_drop=False,
              init_lr=0.001,
              lr_drop=0.5,
              lr_drop_freq=100)

model.compile()
model.train()
