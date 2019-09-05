import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from callback import WarmUpLRSchedule
import json


class Model:
    def __init__(self, core, filepath, pipeline):
        self.core = core
        self.filepath = filepath
        self.pipeline = pipeline
        self.root = os.getcwd()
        self.config = None
        self.callbacks = []
        self.startup()

    @property
    def optimizer(self):
        if self.config['Optimizer']['type'] == 'adam':
            return tf.keras.optimizers.Adam(
                lr=self.config['Optimizer']['lr'],
                beta_1=self.config['Optimizer']['beta_1'],
                beta_2=self.config['Optimizer']['beta_2'])

    @property
    def loss(self):
        return self

    def startup(self):
        self.set_config()
        self.set_callbacks()

    def compile(self):
        self.core.compile(optimizer=self.optimizer, loss=self.loss)
        self.core.summary()

    def train(self):
        self.core.fit_generator(
            generator=self.pipeline.train,
            epochs=self.config['Model']['epoch'],
            initial_epoch=self.config['Model']['ini_epoch'],
            verbose=self.config['Model']['verbose'],
            steps_per_epoch=self.config['Model']['ts_step'],
            callbacks=self.callbacks,
            batch_size=self.config['Model']['batch'],
            validation_data=self.pipeline.valid,
            validation_steps=self.config['Model']['vs_step'])

        self.core.save(self.log_dir + os.sep + 'model.h5')

    def set_callbacks(self):
        if self.config['TensorBoard']['enable']:
            self.callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                write_graph=self.config['TensorBoard']['write_graph']))
        if self.config['LRSchedule']['enable']:
            self.callbacks.append(WarmUpLRSchedule(
                warm_up=self.warm_up,
                schedule=self.schedule))
        if self.config['CheckPoint']['enable']:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=self.log_dir + os.sep + 'checkpoint-{epoch}.h5',
                monitor=self.config['CheckPoint']['monitor'],
                save_weights_only=self.config['CheckPoint']['only_weights'],
                period=self.config['CheckPoint']['period']))
        if self.config['EarlyStop']['enable']:
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=self.config['EarlyStop']['monitor'],
                min_delta=self.config['EarlyStop']['min_delta'],
                patience=self.config['EarlyStop']['min_delta'],
                verbose=self.config['EarlyStop']['verbose'],
                mode=self.config['EarlyStop']['mode']))

    def set_config(self):
        with open(self.filepath) as handle:
            self.config = json.loads(handle.read())

    def warm_up(self, batch):
        lr = self.config['Optimizer']['lr']
        wp_lr = self.config['LRSchedule']['wp_lr']
        wp_step = self.config['LRSchedule']['wp_step']
        if batch < wp_step:
            return wp_lr + (lr - wp_lr) * batch / (wp_step - 1)
        else:
            return lr

    def schedule(self, epoch):
        lr = self.config['Optimizer']['lr']
        epochs = self.config['Model']['epoch']
        if self.config['LRSchedule']['type'] == 'cosine':
            return 0.5 * (1 + np.cos(epoch / epochs * np.pi)) * lr

    @property
    def log_dir(self):
        return (self.root + os.sep + 'logs' +
                os.sep + self.config['Model']['name'])
