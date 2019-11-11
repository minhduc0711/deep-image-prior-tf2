import tensorflow as tf
from tensorflow.keras import models, layers, activations, optimizers
import math


class Conv2DReflect(layers.Layer):
    def __init__(self, filters, kernel_size, padding="SAME", strides=1):
        super(Conv2DReflect, self).__init__()

        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="VALID")

    def build(self, input_shape):
        if self.padding == "SAME":
            _, in_h, in_w, _ = input_shape

            out_h = math.ceil(float(in_h) / float(self.strides[0]))
            out_w = math.ceil(float(in_w) / float(self.strides[1]))

            pad_along_height = max((out_h - 1) * self.strides[0] +
                                   self.kernel_size[0] - in_h, 0)
            pad_along_width = max((out_w - 1) * self.strides[1] +
                                  self.kernel_size[1] - in_w, 0)

            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            self.paddings = tf.constant([[0, 0], [pad_top, pad_bottom],
                                         [pad_left, pad_right], [0, 0]])
        super(Conv2DReflect, self).build(input_shape)

    def call(self, inputs):
        if self.padding == "SAME":
            inputs = tf.pad(inputs, self.paddings, "REFLECT")
        x = self.conv(inputs)
        return x

    def compute_output_shape(self, input_shape):
        if self.padding == "SAME":
            return input_shape
        else:
            N, in_h, in_w, C = input_shape
            out_h = math.ceil(float(in_h) / float(self.strides[0]))
            out_w = math.ceil(float(in_w) / float(self.strides[1]))
            return tf.constant([N, out_w, out_h, C], dtype=tf.int32)


class Downsampler(layers.Layer):
    def __init__(self, filters, kernel_size, scale_factor=2):
        super(Downsampler, self).__init__()

        self.conv1 = Conv2DReflect(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=scale_factor,
                                   padding="SAME")
        self.conv2 = Conv2DReflect(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   padding="SAME")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        return x


class Upsampler(layers.Layer):
    def __init__(self, filters, kernel_size, scale_factor=2,
                 upsampling_mode="bilinear"):
        super(Upsampler, self).__init__()

        self.scale_factor = scale_factor
        self.upsampling_mode = upsampling_mode

        self.conv1 = Conv2DReflect(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   padding="SAME")
        self.conv2 = Conv2DReflect(filters=filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding="SAME")

        self.bn0 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.bn0(inputs)

        x = self.conv1(x)
        x = self.bn1(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.UpSampling2D((self.scale_factor, self.scale_factor),
                                interpolation=self.upsampling_mode)(x)
        return x


class Skip(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Skip, self).__init__()

        self.conv = layers.Conv2D(filters, kernel_size,
                                  strides=1,
                                  padding='SAME')
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x
