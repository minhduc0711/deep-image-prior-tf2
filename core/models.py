import tensorflow as tf
from tensorflow.keras import models, layers
from core.layers import Downsampler, Upsampler, Skip


def skip(input_dim, num_filters_down, ksizes_down,
         num_filters_up, ksizes_up,
         num_filters_skip=None, ksizes_skip=None,
         upsampling_mode="bilinear", sigma_p=1/30,
         resize=None):
    assert len(num_filters_down) == len(
        num_filters_up), "Must have the same number of downsampling & upsampling layers"
    if num_filters_skip:
        assert len(num_filters_skip) == len(
            num_filters_down), "The list of skip layers doesn't have the same length"
    else:
        num_filters_skip = [0] * len(num_filters_down)

    skip_outputs = [None] * len(num_filters_skip)
    model_input = layers.Input(shape=input_dim, dtype=tf.float32)
    x = layers.GaussianNoise(sigma_p)(model_input)

    for i in range(len(num_filters_down)):
        x = Downsampler(num_filters_down[i],
                        ksizes_down[i])(x)
        if num_filters_skip[i]:
            skip_outputs[i] = Skip(num_filters_skip[i], ksizes_skip[i])(x)

    for i in range(len(num_filters_up) - 1, -1, -1):
        if num_filters_skip[i]:
            x = tf.concat((x, skip_outputs[i]), axis=3)
        x = Upsampler(num_filters_up[i],
                      ksizes_up[i],
                      scale_factor=2,
                      upsampling_mode=upsampling_mode)(x)

    # Transform to  3-channel image
    output_image = layers.Conv2D(filters=3, kernel_size=1,
                                 strides=1, padding='SAME',
                                 activation='sigmoid',
                                 name="output_image")(x)
    if resize:
        model_output = tf.image.resize(output_image, resize, method='lanczos3')
    else:
        model_output = output_image
    model = models.Model(inputs=model_input, outputs=model_output)

    return model
