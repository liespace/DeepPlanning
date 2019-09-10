import tensorflow as tf
import numpy as np
import os
from callback import WarmUpLRSchedule


class Model:
    def __init__(self, core, config, pipeline):
        self.core = core
        self.pipeline = pipeline
        self.root = os.getcwd()
        self.config = config
        self.callbacks = []
        self.startup()

    @property
    def optimizer(self):
        if self.config['Optimizer']['type'] == 'adam':
            return tf.keras.optimizers.Adam(
                lr=self.config['Optimizer']['lr'],
                beta_1=self.config['Optimizer']['beta_1'],
                beta_2=self.config['Optimizer']['beta_2'])

    def loss(self, y_true, y_pred):
        print ('hello')
        a_ = self.config['Model']['A']
        b_ = self.config['Model']['B']
        c_ = self.config['Model']['C']
        s_ = self.config['Model']['S']
        batch = self.config['Model']['batch']
        lam0 = self.config['Loss']['lam0']

        loss = 0
        for i in range(batch):
            y_t = y_true[i]
            y_p = y_pred[i]

            uv = tf.cast(y_t[:, 0:2], tf.int64)
            crd_t = y_t[:, -a_:-c_]
            crd_t = tf.keras.backend.repeat_elements(crd_t, b_, axis=-1)
            crd_t = tf.keras.backend.reshape(crd_t, (-1, b_, a_-1))
            candi = tf.gather_nd(y_p, uv)
            candi = tf.reshape(candi, (-1, b_, c_ + a_))
            crd_p = candi[:, :, :a_-1]
            d_crd = tf.reduce_sum(crd_t - tf.sigmoid(crd_p), -1)
            col = tf.argmin(d_crd, axis=-1)
            row = tf.range(tf.shape(col)[-1])
            row = tf.cast(row, tf.int64)
            wh = tf.stack([row[..., tf.newaxis], col[..., tf.newaxis]], axis=-1)

            crd_p = tf.gather_nd(crd_p, wh)
            crd_t = y_t[:, -4:-1]
            crd_t = - tf.log(1. / crd_t - 1.)
            loss_crd = lam0 * tf.reduce_sum(crd_t - crd_p)

            cls_p = candi[:, :, -1]
            cls_p = tf.gather_nd(cls_p, wh)
            class_t = tf.cast(y_t[:, -1], tf.float32)
            loss_cls = tf.keras.backend.binary_crossentropy(
                target=class_t, output=cls_p, from_logits=True)

            obj_p = []
            for b in range(b_):
                obj_p.append(y_p[:, :, b * (a_ + c_) + (a_ - 1)])
            obj_p = tf.stack(obj_p)
            uvc = tf.stack(
                [tf.keras.backend.flatten(uv[:, 0]),
                 tf.keras.backend.flatten(uv[:, 1]), col], axis=-1)
            obj_t = tf.SparseTensor(uvc, [1.0], [s_, s_, b_])
            obj_t = tf.sparse_tensor_to_dense(obj_t)
            loss_obj = tf.keras.backend.binary_crossentropy(
                target=tf.keras.backend.flatten(obj_t),
                output=tf.keras.backend.flatten(obj_p),
                from_logits=True)

            loss += loss_crd + loss_cls + loss_obj
        return loss

    def startup(self):
        self.set_callbacks()

    def compile(self):
        self.core.compile(optimizer=self.optimizer, loss=self.loss)
        self.core.summary()

    def train(self):
        self.core.fit_generator(
            generator=self.pipeline.train,
            steps_per_epoch=self.config['Model']['ts_step'],
            epochs=self.config['Model']['epoch'],
            initial_epoch=self.config['Model']['ini_epoch'],
            verbose=self.config['Model']['verbose'],
            callbacks=self.callbacks,
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
