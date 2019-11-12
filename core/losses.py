import tensorflow.keras.backend as K


def pixelwise_mse(mask=None):
    def loss_fn(y_true, y_pred):
        sqr_err = K.square(y_true - y_pred)
        if mask is not None:
            sqr_err *= mask
        sum_sqr = K.sum(sqr_err, axis=[1, 2, 3])

        return K.mean(sum_sqr, axis=0)
    return loss_fn
