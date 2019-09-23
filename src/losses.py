import tensorflow as tf
from itertools import permutations, product


def DeepWayLoss(config, part='all', log=False):
    a_ = config['Model']['A']
    b_ = config['Model']['B']
    c_ = config['Model']['C']
    batch = config['Model']['batch']
    lam0 = config['Loss']['lam0']
    lam1 = config['Loss']['lam1']
    lam2 = config['Loss']['lam2']

    def dw_cor_metric(y_true, y_pred):
        metric = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, b*(c_+a_):(b+1)*(c_+a_) - 2]
                y_t = y_true[j, :, :, b*(c_+a_):(b+1)*(c_+a_) - 2]
                obj = y_true[j, :, :, (b+1)*(c_+a_) - 2]
                # coord metric
                metric += tf.reduce_sum(tf.keras.backend.abs(
                    y_t - tf.math.sigmoid(y_p))) * obj
        # calculate coord loss
        metric = metric / batch / (a_ - 1)
        if log:
            metric = tf.Print(metric, [metric], message='coord metric: ')
        return metric

    def dw_cla_metric(y_true, y_pred):
        metric = 0
        objs = 0 + tf.keras.backend.epsilon()
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, (b + 1) * (c_ + a_) - 1]
                y_t = y_true[j, :, :, (b + 1) * (c_ + a_) - 1]
                obj = y_true[j, :, :, (b + 1) * (c_ + a_) - 2]
                # class loss
                objs += y_t
                y_p = tf.keras.backend.sigmoid(y_p)
                y_p = tf.cast(y_p > 0.5, y_t.dtype)
                metric += tf.cast(tf.equal(y_p, y_t), y_t.dtype) * y_t
        # calculate class accuracy
        accuracy = metric / objs
        if log:
            accuracy = tf.Print(accuracy, [accuracy], message='class accuracy: ')
        return objs

    def dw_obj_metric(y_true, y_pred):
        metric = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, (b + 1) * (c_ + a_) - 2]
                y_t = y_true[j, :, :, (b + 1) * (c_ + a_) - 2]
                # object loss
                y_p = tf.keras.backend.sigmoid(y_p)
                y_p = tf.cast(y_p > 0.5, y_t.dtype)
                metric += tf.cast(tf.equal(y_p, y_t), y_t.dtype)
        # calculate object accuracy
        accuracy = metric / b_ / batch
        if log:
            accuracy = tf.Print(accuracy, [accuracy], message='object accuracy: ')
        return accuracy

    def dw_cor_loss(y_true, y_pred):
        loss = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, b*(c_+a_):(b+1)*(c_+a_) - 2]
                y_t = y_true[j, :, :, b*(c_+a_):(b+1)*(c_+a_) - 2]
                obj = y_true[j, :, :, (b+1)*(c_+a_) - 2]
                # coord loss
                if config['Loss']['coord'] == 'bce':
                    cor = tf.keras.backend.binary_crossentropy(
                        target=y_t, output=y_p, from_logits=True)
                else:
                    cor = tf.reduce_sum(tf.keras.backend.square(
                        tf.math.log_sigmoid(y_t) - y_p)) * 0.5
                loss += cor * obj
        # calculate coord loss
        loss = lam0 * loss / batch / (a_ - 1)
        if log:
            loss = tf.Print(loss, [loss], message='coord loss: ')
        return loss

    def dw_obj_loss(y_true, y_pred):
        loss = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, (b+1)*(c_+a_) - 2]
                y_t = y_true[j, :, :, (b+1)*(c_+a_) - 2]
                # object loss
                loss += tf.reduce_sum(tf.keras.backend.binary_crossentropy(
                    target=y_t, output=y_p, from_logits=True))
        # calculate object loss
        loss = lam2 * tf.reduce_sum(loss) / b_ / batch
        if log:
            loss = tf.Print(loss, [loss], message='object loss: ')
        return loss

    def dw_cla_loss(y_true, y_pred):
        loss = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, (b+1)*(c_+a_) - 1]
                y_t = y_true[j, :, :, (b+1)*(c_+a_) - 1]
                obj = y_true[j, :, :, (b+1)*(c_+a_) - 2]
                # class loss
                loss += tf.reduce_sum(tf.keras.backend.binary_crossentropy(
                    target=y_t, output=y_p, from_logits=True)) * obj
        # calculate class loss
        loss = lam1 * loss / batch
        if log:
            loss = tf.Print(loss, [loss], message='class loss: ')
        return loss

    def dw_loss(y_true, y_pred):
        cor_loss = dw_cor_loss(y_true, y_pred)
        cla_loss = dw_cla_loss(y_true, y_pred)
        obj_loss = dw_obj_loss(y_true, y_pred)
        return cor_loss + cla_loss + obj_loss

    if part == 'coord':
        return dw_cor_loss
    if part == 'class':
        return dw_cla_loss
    if part == 'object':
        return dw_obj_loss
    if part == 'cor_mt':
        return dw_cor_metric
    if part == 'cla_mt':
        return dw_cla_metric
    if part == 'obj_mt':
        return dw_obj_metric
    else:
        return dw_loss
