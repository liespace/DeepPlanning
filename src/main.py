from dataset import DatasetHolder
from model import Model, GAN
import tensorflow as tf
import cores
import numpy as np

TRAIN_MODE = 'GAN'

if TRAIN_MODE == 'GAN':
    epochs = 60000
    batch_size = 3
    zdim = 0
    dkeeper = DatasetHolder(food_type='gpld', file_type='npz',
                            menu={'in': ['gridmap', 'condition', 'label', 'noise'], 'out': ['yes', 'no', 'en']})
    gkeeper = DatasetHolder(food_type='gpld', file_type='npz',
                            menu={'in': ['gridmap', 'condition', 'noise'], 'out': ['yes']})
    dtrain_set = dkeeper.generator(use_for='train', batch_size=batch_size, zdim=zdim)
    gtrain_set = gkeeper.generator(use_for='train', batch_size=batch_size, zdim=zdim)

    ipu_weight = gkeeper.dir_parent + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model = GAN(name='gan',
                gcore=cores.gcore, dcore=cores.dcore,
                gtrain_set=gtrain_set, dtrain_set=dtrain_set,
                input_shape=gkeeper.gridmap_shape, output_shape=gkeeper.label_shape, ipu_weight=ipu_weight,
                check_period=500, checkpoint=False, tensorboard=True,

                dsteps=5,
                zdim=zdim,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,

                optimizer=tf.keras.optimizers.RMSprop(),
                loss='logcosh',
                metrics=None,

                init_lr=0.001,
                lr_drop=0.5,
                lr_drop_freq=1000.0,
                lr_step_drop=False)

    model.train()

if TRAIN_MODE == 'CNN':
    train_size = 542  # 542
    validation_size = 60
    batch_size = 8
    train_steps = int(np.ceil(train_size / batch_size))
    validation_steps = int(np.ceil(validation_size / batch_size))

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
                  checkpoint=True, check_period=500, save_weights_only=False,
                  tensorboard=True,
                  earlystop=False,

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

    # model.core = tf.keras.models.load_model(model.dir_model, custom_objects={'my_accuracy': keeper.my_accuracy})
    # model.core.summary()
    # model.initial_epoch = 20
    # model.epochs = 30
    # model.train()
