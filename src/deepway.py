from dataset import DatasetHolder
from model import Model, GAN
import tensorflow as tf
import cores
import numpy as np

train_size = 542  # 542
validation_size = 60  # 60
batch_size = 8  # 8
batch_size_vd = 1
train_steps = int(np.ceil(train_size / batch_size))
validation_steps = int(np.ceil(validation_size / batch_size_vd))

keeper = DatasetHolder(food_type='gpld', menu={'in': ['gridmap', 'condition'], 'out': ['delta']})  # ['condition']
train_set = keeper.create_dataset(use_for='train', shuffle_buffer=550, batch_size=batch_size)
validation_set = keeper.create_dataset(use_for='validation', shuffle_buffer=60, batch_size=batch_size)

ipu_weight = keeper.dir_parent + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Model(build_core=cores.beta, name='beta',
              train_set=train_set, validation_set=validation_set,
              input_shape=keeper.gridmap_shape, output_shape=keeper.label_shape,
              ipu_weight=ipu_weight, verbose=1,

              optimizer=tf.keras.optimizers.Adam(),
              loss='logcosh',
              metrics=[tf.keras.metrics.mean_absolute_error, keeper.my_accuracy],
              checkpoint=True, check_period=1, save_weights_only=False,
              tensorboard=True,
              earlystop=False,
              prediction_check=False,

              epochs=60000,
              initial_epoch=0,
              steps_per_epoch=train_steps,
              batch_size=None,
              validation_steps=validation_steps,

              lr_step_drop=False,
              init_lr=0.001,
              lr_drop=0.5,
              lr_drop_freq=100)

model.compile()
model.train()

# checkpoint = '{}/logs/{}/'.format(model.dir_parent, model.name) + 'checkpoint-{20}.h5'
# model.core = tf.keras.models.load_model(checkpoint, custom_objects={'my_accuracy': keeper.my_accuracy})
# model.core.summary()
# model.initial_epoch = 20
# model.epochs = 60000
# model.train()