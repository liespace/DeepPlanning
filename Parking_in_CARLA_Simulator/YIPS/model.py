from __future__ import print_function
import os
import logging
import numpy as np
import tensorflow as tf
from .cores import Core
from .dataset import Pipeline
from .losses import DeepWayLoss
from .callback import WarmUpLRSchedule


class DWModel(object):
    def __init__(self, config):
        self.pipeline = Pipeline(config)
        self.model = Core(config)
        self.config = config
        self.callbacks = []
        self.startup()

    @property
    def optimizer(self):
        if self.config['Optimizer']['type'] == 'adam':
            tf.logging.warning('Using Optimizer Adam')
            return tf.keras.optimizers.Adam(
                lr=self.config['Optimizer']['lr'],
                beta_1=self.config['Optimizer']['beta_1'],
                beta_2=self.config['Optimizer']['beta_2'])
        if self.config['Optimizer']['type'] == 'sgd':
            tf.logging.warning('Using Optimizer SGD')
            return tf.keras.optimizers.SGD(
                lr=self.config['Optimizer']['lr'],
                momentum=self.config['Optimizer']['momentum'],
                decay=self.config['Optimizer']['decay'])

    def startup(self):
        self.set_callbacks()

    def compile(self):
        self.model.core.compile(
            optimizer=self.optimizer,
            loss=DeepWayLoss(self.config),
            metrics=[DeepWayLoss(self.config, 'coord'),
                     DeepWayLoss(self.config, 'object'),
                     DeepWayLoss(self.config, 'cor_mt'),
                     DeepWayLoss(self.config, 'obj_mt')])
        if self.config['Train']['summary']:
            self.model.core.summary()
        return self

    def freeze_backbone(self):
        tf.logging.warning('Freezing backbone')
        for layer in self.model.backbone.layers:
            layer.trainable = False
        self.compile()

    def unfreeze_backbone(self):
        tf.logging.warning('Unfreezing backbone')
        for layer in self.model.backbone.layers:
            layer.trainable = True
        self.compile()

    def train(self):
        # settings
        batch = int(self.config['Train']['batch'])
        training_set_size = int(self.config['Train']['training_set_size'])
        validation_set_size = int(self.config['Train']['validation_set_size'])
        training_steps = int(np.ceil(float(training_set_size) / float(batch)))
        validation_steps = int(np.ceil(float(validation_set_size) / float(batch)))
        # flowing
        self.freeze_backbone()
        self.model.core.fit_generator(
            generator=self.pipeline.train,
            steps_per_epoch=training_steps,
            epochs=self.config['Train']['frozen_epoch'],
            initial_epoch=self.config['Train']['ini_epoch'],
            verbose=self.config['Train']['verbose'],
            callbacks=self.callbacks,
            validation_data=self.pipeline.valid,
            validation_steps=validation_steps,
            max_queue_size=self.config['Train']['max_queue_size'])
        self.model.core.save(self.log_dir + os.sep + 'fr-model.h5')

        self.unfreeze_backbone()
        self.model.core.fit_generator(
            generator=self.pipeline.train,
            steps_per_epoch=training_steps,
            epochs=self.config['Train']['epoch'],
            initial_epoch=self.config['Train']['frozen_epoch'],
            verbose=self.config['Train']['verbose'],
            callbacks=self.callbacks,
            validation_data=self.pipeline.valid,
            validation_steps=validation_steps,
            max_queue_size=self.config['Train']['max_queue_size'])
        self.model.core.save(self.log_dir + os.sep + 'model.h5')

    def load_weights(self, weights_filepath):
        self.model.core.load_weights(filepath=weights_filepath)
        logging.info('Loading weights from ' + weights_filepath)
        return self

    def predict(self, samples):
        return self.model.core.predict(x=samples, batch_size=1, verbose=1)

    def predict_on_sample(self, x):
        return self.model.core.predict_on_batch(x=x)

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
        epochs = self.config['Train']['epoch']
        if self.config['LRSchedule']['type'] == 'cosine':
            lrd = 0.5 * (1.0 + np.cos(epoch / float(epochs) * np.pi)) * lr
            return lrd
        if self.config['LRSchedule']['type'] == 'steps':
            drop_epochs, drop_decay = self.config['LRSchedule']['drop_epochs'], self.config['LRSchedule']['drop_decay']
            for ep in drop_epochs:
                lr /= drop_decay if epoch >= ep else 1.
            # print ('decay', lr)
            return lr
        return lr

    @property
    def log_dir(self):
        log_folder = '{}-{}{}-(b{})-({}_{:1.0e}_{:1.0e})-({}_{:1.0e})-(fr{}_{}{}{}_wp{}o{:1.0e})'.format(
            self.config['Train']['inputs_type'],
            self.config['Model']['name'], 'T' if self.config['Model']['tiny'] else 'C',
            self.config['Train']['batch'],
            self.config['Loss']['coord'], self.config['Loss']['lam0'], self.config['Loss']['lamb'],
            self.config['Optimizer']['type'], self.config['Optimizer']['lr'],
            self.config['Train']['frozen_epoch'], self.config['LRSchedule']['type'],
            int(self.config['LRSchedule']['drop_decay']), self.config['LRSchedule']['drop_epochs'],
            self.config['LRSchedule']['wp_step'], self.config['LRSchedule']['wp_lr']
        )
        return os.getcwd() + os.sep + 'logs' + os.sep + log_folder
