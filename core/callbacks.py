import tensorflow as tf
from tensorflow.keras import callbacks, models

import os
import imageio


class SaveResultImage(callbacks.Callback):
    def __init__(self, input_tensor, output_dir, img_name, n=500):
        """
        Callback to save the output of model or intermediate layer 
        every n epochs. This code is a little dumb as the model has 
        to run another forward pass

        Params:
            input_tensor (Tensor): input to the model
            output_dir (str): where the image will be saved
            img_name (str): the filename will be img_name + epoch.jpg
            n: number of epochs in between saves
        """
        super(SaveResultImage, self).__init__()
        self.input_tensor = input_tensor
        self.n = n
        self.output_dir = output_dir
        self.img_name = img_name

    def on_epoch_end(self, epoch, logs=None):
        if (epoch == 0) or ((epoch + 1) % self.n == 0):
            print("Saving output image...")

            intermediate_model = models.Model(inputs=self.model.input,
                                              outputs=self.model.get_layer("output_image").output)
            y_pred = intermediate_model(self.input_tensor, training=True)
            output_img = tf.image.convert_image_dtype(tf.squeeze(y_pred),
                                                      dtype=tf.uint8).numpy()
            output_path = os.path.join(self.output_dir,
                                       f"{self.img_name}_{epoch + 1}.jpg")
            imageio.imsave(output_path, output_img)
