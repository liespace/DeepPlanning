import tensorflow as tf
from itertools import permutations, product


def DeepWayLoss(config, log=False):
    a_ = config['Model']['A']
    b_ = config['Model']['B']
    c_ = config['Model']['C']
    s_ = config['Model']['S']
    m_ = config['Model']['M']
    batch = config['Model']['batch']
    lam0 = config['Loss']['lam0']
    lam1 = config['Loss']['lam1']

    def dw_loss(y_true, y_pred):
        for j in range(batch):
            ls_cor, ls_cla, obj_ts = [], [], []
            for m in range(m_):
                l_cor, l_cla, obj_tc = [], [], []
                y_t = y_true[j, :, :, m*(c_+a_):(m+1)*(c_+a_)]
                mask = y_t[:, :, a_ - 1]
                for i in range(b_):
                    y_p = y_pred[j, :, :, i*(c_+a_):(i+1)*(c_+a_)]
                    # coord loss
                    cor_p = y_p[:, :, :a_-1] * tf.stack([mask] * (a_-1), axis=-1)
                    cor_t = y_t[:, :, :a_-1]
                    l_cor.append(tf.reduce_sum(tf.abs(cor_t - cor_p)))
                    # class loss
                    cla_p = y_p[:, :, a_] * mask
                    cla_t = y_t[:, :, a_]
                    l_cla.append(tf.keras.backend.binary_crossentropy(
                        target=tf.keras.backend.flatten(cla_t),
                        output=tf.keras.backend.flatten(cla_p)))
                    # object true candidates
                    can = [tf.keras.backend.zeros((s_, s_)) for b in range(b_)]
                    can[i] = mask
                    obj_tc.append(tf.keras.backend.stack(can, axis=-1))
                arg = tf.keras.backend.argmin(l_cor, axis=-1)
                ls_cor.append(tf.gather(l_cor, arg))
                ls_cla.append(tf.gather(l_cla, arg))
                obj_ts.append(tf.gather(obj_tc, arg))
            # calculate object loss
            obj_ts = tf.stack(obj_ts, axis=-1)
            obj_t = tf.reduce_sum(obj_ts, axis=-1)
            obj_p = []
            for b in range(b_):
                y_p = y_pred[j, :, :, b * (c_ + a_):(b + 1) * (c_ + a_)]
                obj_p.append(y_p[:, :, a_-1])
            obj_p = tf.stack(obj_p, axis=-1)
            ls_obj = tf.keras.backend.binary_crossentropy(obj_t, obj_p)
            ls_obj = tf.reduce_sum(ls_obj) * (1.0 / (s_ * s_ * b_))
            # calculate overall loss
            ls_cor = lam0 * tf.reduce_sum(ls_cor)
            ls_cla = lam1 * tf.reduce_sum(ls_cla)
            loss = ls_cor + ls_cla + ls_obj
            if log:
                loss = tf.Print(loss, [loss, ls_cor, ls_cla, ls_obj],
                                message='loss: ')
            return loss

    return dw_loss
