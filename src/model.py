from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from callback import WarmUpLRSchedule
from losses import DeepWayLoss


class DWModel:
    def __init__(self, core, config, pipeline):
        self.model = core
        self.pipeline = pipeline
        self.root = os.getcwd()
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

    def compile(self, summary=True):
        self.model.core.compile(
            optimizer=self.optimizer,
            loss=DeepWayLoss(self.config),
            metrics=[DeepWayLoss(self.config, 'coord'),
                     DeepWayLoss(self.config, 'object'),
                     DeepWayLoss(self.config, 'cor_mt'),
                     DeepWayLoss(self.config, 'obj_mt')])
        if summary:
            self.model.core.summary()

    def freeze_backbone(self):
        tf.logging.warning('Freezing backbone')
        for layer in self.model.backbone.layers:
            layer.trainable = False
        self.compile(summary=True)

    def unfreeze_backbone(self):
        tf.logging.warning('Unfreezing backbone')
        for layer in self.model.backbone.layers:
            layer.trainable = True
        self.compile(summary=True)

    def train(self):
        # settings
        batch = int(self.config['Train']['batch'])
        ts_size = int(self.config['Train']['ts_size'])
        vs_size = int(self.config['Train']['vs_size'])
        t_step = int(np.ceil(float(ts_size) / float(batch)))
        v_step = int(np.ceil(float(vs_size) / float(batch)))
        # flowing
        self.freeze_backbone()
        self.model.core.fit_generator(
            generator=self.pipeline.train,
            steps_per_epoch=t_step,
            epochs=self.config['Train']['frozen_epoch'],
            initial_epoch=self.config['Train']['ini_epoch'],
            verbose=self.config['Train']['verbose'],
            callbacks=self.callbacks,
            validation_data=self.pipeline.valid,
            validation_steps=v_step,
            max_queue_size=self.config['Train']['max_queue_size'])
        self.model.core.save(self.log_dir + os.sep + 'fr-model.h5')

        self.unfreeze_backbone()
        self.model.core.fit_generator(
            generator=self.pipeline.train,
            steps_per_epoch=t_step,
            epochs=self.config['Train']['epoch'],
            initial_epoch=self.config['Train']['frozen_epoch'],
            verbose=self.config['Train']['verbose'],
            callbacks=self.callbacks,
            validation_data=self.pipeline.valid,
            validation_steps=v_step,
            max_queue_size=self.config['Train']['max_queue_size'])
        self.model.core.save(self.log_dir + os.sep + 'model.h5')

    def predict_generator(self, weights_file):
        self.model.core.load_weights(filepath=weights_file)
        print('loading conditions from ' + weights_file)
        return self.model.core.predict_generator(
            generator=self.pipeline.cond,
            steps=int(self.config['Pred']['pd_size']),
            verbose=self.config['Pred']['verbose'],
            max_queue_size=self.config['Train']['max_queue_size'])

    def predict(self, x):
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
        if self.config['LRSchedule']['type'] == 'unfrozen':
            if epoch >= self.config['Train']['frozen_epoch']:
                lrd = lr / self.config['LRSchedule']['unfrozen_decay']
                print ('decay', lrd)
                return lrd
        return lr

    @property
    def log_dir(self):
        return (self.root + os.sep + 'logs' +
                os.sep + self.config['Model']['name'])

    # def wrapper(self):
    #     a_ = self.config['Model']['A']
    #     b_ = self.config['Model']['B']
    #     c_ = self.config['Model']['C']
    #     s_ = self.config['Model']['S']
    #     y_true = tf.keras.layers.Input(shape=(s_, s_, b_ * (a_ + c_)))
    #     loss = tf.keras.layers.Lambda(
    #         self.dw_loss, output_shape=(1,), name='dw_loss',
    #         arguments={'config': self.config})(
    #         [self.core.output, y_true])
    #     self.model = tf.keras.Model([self.core.input, y_true], loss)
