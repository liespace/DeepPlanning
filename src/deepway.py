from dataset import Pipeline
from model import Model
import tensorflow as tf
import cores
import numpy as np

train_size = 542  # 542
validation_size = 60  # 60
batch_size = 8  # 8
batch_size_vd = 1
train_steps = int(np.ceil(train_size / batch_size))
validation_steps = int(np.ceil(validation_size / batch_size_vd))

pipeline = Pipeline()

fen_ws = pipeline.root + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Model(build_core=cores.beta, name='beta',
              train_set=train_set, valid_set=validation_set,
              input_shape=keeper.gridmap_shape, output_shape=keeper.label_shape,
              backbone_ws=fen_ws, verbose=1,

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