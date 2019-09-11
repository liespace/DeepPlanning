import tensorflow as tf
from itertools import permutations, product


def DeepWayLoss(config, part='all', log=False):
    a_ = config['Model']['A']
    b_ = config['Model']['B']
    c_ = config['Model']['C']
    s_ = config['Model']['S']
    m_ = config['Model']['M']
    batch = config['Model']['batch']
    lam0 = config['Loss']['lam0']
    lam1 = config['Loss']['lam1']
    lam2 = config['Loss']['lam2']

    def dw_cor_loss(y_true, y_pred):
        for j in range(batch):
            losses = []
            for m in range(m_):
                candi = []
                y_t = y_true[j, :, :, m*(c_+a_):(m+1)*(c_+a_)]
                mask = y_t[:, :, a_ - 1]
                for i in range(b_):
                    y_p = y_pred[j, :, :, i*(c_+a_):(i+1)*(c_+a_)]
                    # coord loss
                    cor_p = y_p[:, :, :a_-1] * tf.stack([mask] * (a_-1), axis=-1)
                    cor_t = y_t[:, :, :a_-1]
                    candi.append(tf.reduce_sum(
                        tf.keras.backend.binary_crossentropy(cor_t, cor_p)))
                # find key index
                arg = tf.keras.backend.argmin(candi, axis=-1)
                # store the selected candidate loss
                losses.append(tf.gather(candi, arg))
            # calculate coord loss
            loss = (lam0 / (a_ - 1)) * tf.reduce_sum(losses)
            if log:
                loss = tf.Print(loss, [loss], message='coord loss: ')
            return loss

    def dw_cla_loss(y_true, y_pred):
        for j in range(batch):
            losses = []
            for m in range(m_):
                l_cor, candi = [], []
                y_t = y_true[j, :, :, m*(c_+a_):(m+1)*(c_+a_)]
                mask = y_t[:, :, a_ - 1]
                for i in range(b_):
                    y_p = y_pred[j, :, :, i*(c_+a_):(i+1)*(c_+a_)]
                    # coord loss
                    cor_p = y_p[:, :, :a_-1] * tf.stack([mask] * (a_-1), axis=-1)
                    cor_t = y_t[:, :, :a_-1]
                    l_cor.append(tf.reduce_sum(
                        tf.keras.backend.binary_crossentropy(cor_t, cor_p)))
                    # class loss
                    cla_p = y_p[:, :, a_] * mask
                    cla_t = y_t[:, :, a_]
                    candi.append(tf.keras.backend.binary_crossentropy(
                        target=cla_t, output=cla_p))
                # find key index
                arg = tf.keras.backend.argmin(l_cor, axis=-1)
                # store the selected candidate loss
                losses.append(tf.gather(candi, arg))
            # calculate class loss
            loss = lam1 * tf.reduce_sum(losses)
            if log:
                loss = tf.Print(loss, [loss], message='class loss: ')
            return loss

    def dw_obj_loss(y_true, y_pred):
        for j in range(batch):
            grounds = []
            for m in range(m_):
                l_cor, obj_tc = [], []
                y_t = y_true[j, :, :, m*(c_+a_):(m+1)*(c_+a_)]
                mask = y_t[:, :, a_ - 1]
                for i in range(b_):
                    y_p = y_pred[j, :, :, i*(c_+a_):(i+1)*(c_+a_)]
                    # coord loss
                    cor_p = y_p[:, :, :a_-1] * tf.stack([mask] * (a_-1), axis=-1)
                    cor_t = y_t[:, :, :a_-1]
                    l_cor.append(tf.reduce_sum(
                        tf.keras.backend.binary_crossentropy(cor_t, cor_p)))
                    # object true candidates
                    can = [tf.keras.backend.zeros((s_, s_)) for b in range(b_)]
                    can[i] = mask
                    obj_tc.append(tf.keras.backend.stack(can, axis=-1))
                # find key index
                arg = tf.keras.backend.argmin(l_cor, axis=-1)
                # store the ground true candidates
                grounds.append(tf.gather(obj_tc, arg))
            # calculate object loss
            grounds = tf.stack(grounds, axis=-1)
            obj_t = tf.reduce_sum(grounds, axis=-1)
            obj_p = []
            for b in range(b_):
                y_p = y_pred[j, :, :, b * (c_ + a_):(b + 1) * (c_ + a_)]
                obj_p.append(y_p[:, :, a_-1])
            obj_p = tf.stack(obj_p, axis=-1)
            loss = tf.keras.backend.binary_crossentropy(obj_t, obj_p)
            loss = lam2 * tf.reduce_mean(loss)
            if log:
                loss = tf.Print(loss, [loss], message='object loss: ')
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
    else:
        return dw_loss
