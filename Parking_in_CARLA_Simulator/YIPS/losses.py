import tensorflow as tf


def DeepWayLoss(config, part='all', log=False):
    a_ = config['Model']['A']
    b_ = config['Model']['B']
    batch = config['Train']['batch']
    lam0 = config['Loss']['lam0']
    lam1 = config['Loss']['lam1']

    def dw_cor_metric(y_true, y_pred):
        metric = 0
        num = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, b*a_:(b+1)*a_ - 1]
                y_t = y_true[j, :, :, b*a_:(b+1)*a_ - 1]
                obj = y_true[j, :, :, (b+1)*a_ - 1]
                # coord metric
                metric += tf.reduce_sum(tf.keras.backend.abs(
                    y_t - tf.math.sigmoid(y_p))) * obj
                num += obj
        # calculate coord loss
        metric = metric / (a_ - 1) / (num + tf.keras.backend.epsilon())
        if log:
            metric = tf.Print(metric, [metric], message='coord metric: ')
        return metric

    def dw_obj_metric(y_true, y_pred):
        metric = 0
        for j in range(batch):
            flag = 0
            for b in range(b_):
                y_p = y_pred[j, :, :, (b + 1) * a_ - 1]
                y_t = y_true[j, :, :, (b + 1) * a_ - 1]
                # object loss
                y_p = tf.keras.backend.sigmoid(y_p)
                y_p = tf.cast(y_p > 0.5, y_t.dtype)
                flag += tf.cast(tf.equal(y_p, y_t), y_t.dtype)
            metric += tf.cast(tf.equal(flag, b_), tf.float16)
        # calculate object accuracy
        accuracy = metric / batch
        if log:
            accuracy = tf.Print(accuracy, [accuracy], message='object accuracy: ')
        return accuracy

    def dw_cor_loss(y_true, y_pred):
        loss = 0
        num = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, b*a_:(b+1)*a_ - 1]
                y_t = y_true[j, :, :, b*a_:(b+1)*a_ - 1]
                obj = y_true[j, :, :, (b+1)*a_ - 1]
                # coord loss
                if config['Loss']['coord'] == 'bce':
                    cor = tf.keras.backend.binary_crossentropy(
                        target=y_t, output=y_p, from_logits=True)
                else:
                    cor = tf.reduce_sum(tf.keras.backend.square(
                        tf.math.log((y_t+tf.keras.backend.epsilon())/(1.-y_t+tf.keras.backend.epsilon())) - y_p)) * 0.5
                loss += cor * obj
                num += obj
        # calculate coord loss
        loss = lam0 * loss / (a_ - 1) / (num + tf.keras.backend.epsilon())
        if log:
            loss = tf.Print(loss, [loss], message='coord loss: ')
        return loss

    def dw_obj_loss(y_true, y_pred):
        loss = 0
        for j in range(batch):
            for b in range(b_):
                y_p = y_pred[j, :, :, (b+1)*a_ - 1]
                y_t = y_true[j, :, :, (b+1)*a_ - 1]
                # object loss
                loss += tf.reduce_sum(tf.keras.backend.binary_crossentropy(
                    target=y_t, output=y_p, from_logits=True))
        # calculate object loss
        loss = lam1 * tf.reduce_sum(loss) / b_ / batch
        if log:
            loss = tf.Print(loss, [loss], message='object loss: ')
        return loss

    def dw_loss(y_true, y_pred):
        cor_loss = dw_cor_loss(y_true, y_pred)
        obj_loss = dw_obj_loss(y_true, y_pred)
        return cor_loss + obj_loss

    if part == 'coord':
        return dw_cor_loss
    if part == 'object':
        return dw_obj_loss
    if part == 'cor_mt':
        return dw_cor_metric
    if part == 'obj_mt':
        return dw_obj_metric
    else:
        return dw_loss
