import tensorflow as tf
from itertools import permutations, product


def DeepWayLoss(config, simple=True):
    a_ = config['Model']['A']
    b_ = config['Model']['B']
    c_ = config['Model']['C']
    s_ = config['Model']['S']
    batch = config['Model']['batch']
    lam0 = config['Loss']['lam0']
    lam1 = config['Loss']['lam1']
    if not simple:
        pm = list(permutations(range(b_)))
        combo = list(product(pm, pm))
    else:
        combo = [(range(b_), range(b_))]

    def work(y_true, y_pred):
        loss_crd, loss_cla, loss_obj = 0., 0., 0.
        m_loss, m_crd, m_cla, m_obj = 1e10, 1e10, 1e10, 1e10
        for j in range(batch):
            for pair in combo:
                ti, pi = pair[0], pair[1]
                for i in range(b_):
                    y_p = y_pred[j, :, :, pi[i] * (c_ + a_):(pi[i] + 1) * (c_ + a_)]
                    y_t = y_true[j, :, :, ti[i] * (c_ + a_):(ti[i] + 1) * (c_ + a_)]
                    mask = y_t[:, :, a_ - 1]
                    crd_p = y_p[:, :, :a_ - 1] * tf.stack([mask] * (a_ - 1),
                                                          axis=-1)
                    crd_t = y_t[:, :, :a_ - 1]
                    loss_crd += tf.reduce_sum(tf.abs(crd_t - crd_p))

                    cla_p = y_p[:, :, a_] * mask
                    cla_t = y_t[:, :, a_]
                    l_cla = tf.keras.backend.binary_crossentropy(
                        target=tf.keras.backend.flatten(cla_t),
                        output=tf.keras.backend.flatten(cla_p))
                    loss_cla += tf.keras.backend.sum(l_cla)

                    obj_p = y_p[:, :, a_ - 1]
                    obj_t = y_t[:, :, a_ - 1]
                    l_obj = tf.keras.backend.binary_crossentropy(
                        target=tf.keras.backend.flatten(obj_t),
                        output=tf.keras.backend.flatten(obj_p))
                    loss_obj += tf.keras.backend.sum(l_obj)

                loss_crd *= lam0
                loss_cla *= lam1
                loss_obj *= (1.0 / (s_ * s_ * b_))
                loss = loss_crd + loss_cla + loss_obj
                m_loss = tf.keras.backend.minimum(m_loss, loss)
                m_crd = tf.keras.backend.minimum(m_crd, loss_crd)
                m_cla = tf.keras.backend.minimum(m_cla, loss_cla)
                m_obj = tf.keras.backend.minimum(m_obj, loss_obj)
            m_loss = tf.Print(m_loss, [m_loss, m_crd, m_cla, m_obj],
                              message='loss: ')
        return m_loss
    return work
