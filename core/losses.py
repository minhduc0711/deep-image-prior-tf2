import tensorflow as tf


def pixelwise_mse(y_true, y_pred):
    batch_size = y_true.shape[0]
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    return tf.keras.losses.mean_squared_error(y_true, y_pred)
