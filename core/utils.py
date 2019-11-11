import tensorflow as tf


def crop_div_32(img):
    h, w = img.shape[:2]
    new_h = h - (h % 32)
    new_w = w - (w % 32)
    return tf.image.resize_with_crop_or_pad(img, new_h, new_w).numpy()
