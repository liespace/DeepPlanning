import tensorflow as tf


def SVG16(x, l2_reg=0.0005):
    conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv1_1')(x)
    conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name='conv5_3')(conv5_2)
    return conv5_3


if __name__ == '__main__':
    inputs = tf.keras.layers.Input(shape=(480, 480, 3), dtype=tf.float32)
    svg16 = tf.keras.Model(inputs=inputs, outputs=SVG16(inputs))
    svg16.load_weights('VGG_VOC0712_SSD_512x512_iter_120000.h5', by_name=True)
    svg16.summary()
