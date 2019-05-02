import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from functools import partial
from callback import PredictionCheckPoint


class Model:
    def __init__(self, name='main', build_core=None,
                 train_set=None, validation_set=None,
                 input_shape=None, output_shape=None, ipu_weight=None,
                 checkpoint=True, check_period=100, save_weights_only=True, tensorboard=True, prediction_check=False,
                 earlystop=False, monitor='loss', min_delta=0.001, patience=100, mode='auto', baseline=None,
                 optimizer=tf.keras.optimizers.Adam(),
                 loss='logcosh',
                 metrics=None,
                 epochs=4,
                 initial_epoch=0,
                 verbose=1,
                 steps_per_epoch=100,
                 batch_size=None,
                 validation_steps=10,
                 init_lr=0.001,
                 lr_drop=0.5,
                 lr_drop_freq=1000.0,
                 lr_step_drop=False):
        self.name = name
        self.build_core = build_core
        self.train_set = train_set
        self.validation_set = validation_set
        self.ipu_weight = ipu_weight

        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.validation_steps = validation_steps
        self.verbose = verbose
        self.init_lr = init_lr
        self.lr_drop = lr_drop
        self.lr_drop_freq = lr_drop_freq

        self.core = None
        self.output = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.ipu = tf.keras.applications.ResNet50(input_shape=self.input_shape, weights=None, include_top=False)
        self.ipu.load_weights(self.ipu_weight)
        self.oru = tf.keras.Sequential()
        self.oru.add(tf.keras.layers.Reshape(target_shape=self.output_shape, input_shape=(np.prod(self.output_shape),)))
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss

        self.dir_parent = os.path.dirname(os.getcwd())
        self.dir_log = '{}/logs/{}/'.format(self.dir_parent, self.name) + self.get_now_str()
        self.dir_checkpoint = '{}/logs/{}/'.format(self.dir_parent, self.name) + 'checkpoint-{epoch}.h5'
        self.dir_model = '{}/logs/{}/'.format(self.dir_parent, self.name) + '{}.h5'.format(self.name)
        self.check_period = check_period
        self.lr_step_drop = lr_step_drop
        self.checkpoint = checkpoint
        self.tensorboard = tensorboard
        self.checkpoint = checkpoint
        self.earlystop = earlystop
        self.prediction_check = prediction_check
        self.callbacks = []

        self.save_weights_only = save_weights_only

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        if self.tensorboard:
            self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.dir_log, write_graph=False))
        if self.lr_step_drop:
            self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=self.step_decay))
        if self.checkpoint:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.dir_checkpoint, monitor='loss',
                                                                     save_weights_only=self.save_weights_only,
                                                                     period=self.check_period))
        if self.earlystop:
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=self.monitor, min_delta=self.min_delta,
                                                                   patience=self.patience, verbose=self.verbose,
                                                                   mode=self.mode, baseline=self.baseline))
        if self.prediction_check:
            self.callbacks.append(PredictionCheckPoint(dir_parent=self.dir_parent))

    def compile(self):
        self.core = self.build_core(self.input_shape, self.output_shape, self.ipu, self.oru)
        self.core.compile(optimizer=tf.keras.optimizers.Adam(), loss=self.loss, metrics=self.metrics)
        self.core.summary()

    def train(self):
        self.core.fit(self.train_set,
                      epochs=self.epochs,
                      initial_epoch=self.initial_epoch,
                      verbose=self.verbose,
                      steps_per_epoch=self.steps_per_epoch,
                      callbacks=self.callbacks,
                      batch_size=self.batch_size,
                      validation_data=self.validation_set,
                      validation_steps=self.validation_steps)

        self.core.save(self.dir_model)

    def step_decay(self, epoch):
        return self.init_lr * np.power(self.lr_drop, np.floor((1 + epoch) / self.lr_drop_freq))

    @staticmethod
    def get_now_str():
        return str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')


class GAN(Model):
    def __init__(self, name='gan',
                 gcore=None, dcore=None,
                 gtrain_set=None, dtrain_set=None,
                 input_shape=None, output_shape=None, ipu_weight=None,
                 check_period=100, checkpoint=True, tensorboard=True,
                 dsteps=5,
                 zdim=100,
                 epochs=4,
                 batch_size=2,
                 verbose=1,
                 optimizer=tf.keras.optimizers.RMSprop(lr=0.00005),
                 loss='logcosh',
                 metrics=None,
                 init_lr=0.001,
                 lr_drop=0.5,
                 lr_drop_freq=1000.0,
                 lr_step_drop=False):
        super().__init__(name=name, input_shape=input_shape, output_shape=output_shape, ipu_weight=ipu_weight,
                         check_period=check_period, checkpoint=checkpoint, tensorboard=tensorboard, verbose=verbose,
                         optimizer=optimizer, loss=loss, metrics=metrics, epochs=epochs, batch_size=batch_size,
                         init_lr=init_lr, lr_drop=lr_drop, lr_drop_freq=lr_drop_freq, lr_step_drop=lr_step_drop)
        # Build the generator and critic
        self.dtrain_set = dtrain_set
        self.gtrain_set = gtrain_set
        self.gcore = gcore(input_shape, output_shape, zdim, self.ipu, self.oru)
        self.dcore = dcore(input_shape, output_shape, self.ipu)
        self.zdim = zdim

        self.dir_dmodel = '{}/logs/{}/'.format(self.dir_parent, self.name) + '{}.h5'.format('dmodel')
        self.dir_gmodel = '{}/logs/{}/'.format(self.dir_parent, self.name) + '{}.h5'.format('gmodel')

        self.dsteps = dsteps
        self.dmodel = None
        self.gmodel = None
        self.pack_discriminator()
        self.pack_generator()

        timestamp = self.get_now_str()
        self.dir_dlog = '{}/logs/{}/'.format(self.dir_parent, self.name) + timestamp + '/d/'
        self.dir_glog = '{}/logs/{}/'.format(self.dir_parent, self.name) + timestamp + '/g/'
        self.dcallbacks = []
        self.gcallbacks = []
        if self.tensorboard:
            self.dcallbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.dir_dlog, write_graph=False))
            self.gcallbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.dir_glog, write_graph=False))
        if self.lr_step_drop:
            self.dcallbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=self.step_decay))
            self.gcallbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule=self.step_decay))
        if self.checkpoint:
            self.dcallbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.dir_checkpoint, monitor='loss',
                                                                      save_weights_only=self.save_weights_only,
                                                                      period=self.check_period))
            self.gcallbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.dir_checkpoint, monitor='loss',
                                                                      save_weights_only=self.save_weights_only,
                                                                      period=self.check_period))
        if self.earlystop:
            self.dcallbacks.append(tf.keras.callbacks.EarlyStopping(monitor=self.monitor, min_delta=self.min_delta,
                                                                    patience=self.patience, verbose=self.verbose,
                                                                    mode=self.mode, baseline=self.baseline))
            self.gcallbacks.append(tf.keras.callbacks.EarlyStopping(monitor=self.monitor, min_delta=self.min_delta,
                                                                    patience=self.patience, verbose=self.verbose,
                                                                    mode=self.mode, baseline=self.baseline))

    def pack_discriminator(self):
        # Freeze generator's layers while training critic
        self.gcore.trainable = False
        self.dcore.trainable = True

        gridmap = tf.keras.layers.Input(shape=self.input_shape)
        cdt = tf.keras.layers.Input(shape=self.output_shape)
        real = tf.keras.layers.Input(shape=self.output_shape)
        z_disc = tf.keras.layers.Input(shape=(self.zdim,))

        yes = self.dcore([gridmap, cdt, real])
        fake = self.gcore([gridmap, cdt, z_disc])
        no = self.dcore([gridmap, cdt, fake])

        # Construct weighted average between real and fake images
        mid = tf.keras.layers.Lambda(self.random_weighted_average)([real, fake])
        # Determine validity of weighted sample
        en = self.dcore([gridmap, cdt, mid])  # validity_interpolated

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=mid)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.dmodel = tf.keras.Model(inputs=[gridmap, cdt, real, z_disc], outputs=[yes, no, en])
        self.dmodel.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
                            optimizer=self.optimizer,
                            loss_weights=[1, 1, 10])
        self.dmodel.summary()

    def random_weighted_average(self, inputs):
        alpha = tf.keras.backend.random_uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def pack_generator(self):
        # For the generator we freeze the critic's layers
        self.dcore.trainable = False
        self.gcore.trainable = True

        gridmap = tf.keras.layers.Input(shape=self.input_shape)
        cdt = tf.keras.layers.Input(shape=self.output_shape)
        z_gen = tf.keras.layers.Input(shape=(self.zdim,))

        case = self.gcore([gridmap, cdt, z_gen])
        judgement = self.dcore([gridmap, cdt, case])

        self.gmodel = tf.keras.Model([gridmap, cdt, z_gen], judgement)
        self.gmodel.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
        self.gmodel.summary()

    @staticmethod
    def gradient_penalty_loss(y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = tf.keras.backend.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = tf.keras.backend.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = tf.keras.backend.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = tf.keras.backend.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = tf.keras.backend.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return tf.keras.backend.mean(gradient_penalty)

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def train(self):
        # TODO Rescale -1 to 1
        for epoch in range(self.epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_loss = self.dmodel.fit_generator(self.dtrain_set, epochs=epoch + 1, initial_epoch=epoch,
                                               verbose=self.verbose,
                                               steps_per_epoch=self.dsteps,
                                               callbacks=self.dcallbacks)

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.gmodel.fit_generator(self.gtrain_set, epochs=epoch + 1, initial_epoch=epoch,
                                               verbose=self.verbose,
                                               steps_per_epoch=1,
                                               callbacks=self.gcallbacks)
            if epoch % self.check_period == 0:
                self.dmodel.save(self.dir_dmodel)
                self.gmodel.save(self.dir_gmodel)
