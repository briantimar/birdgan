""" tf GAN models """
import tensorflow as tf

class DCGAN:
    """DCGAN:  https://arxiv.org/pdf/1511.06434.pdf.
    Here I'm borrowing the implementation from the TF tutorial (see readme)"""

    def __init__(self):
        pass

    def build_generator(self, input_shape, output_shape=(64,64)):
        """ build image generator based on sampling from noise
        tensor of given shape."""

        #how many conv layers to apply in upsampling to output
        num_upsample=2
        #how many channels to start with
        init_channels=256
        #pick strides to end up with correct output shape. Will be doubling
        #at each transpose-conv
        init_size = output_shape[0]//(2**num_upsample)
        kernel_size=(5,5)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(init_size * init_size * init_channels,
                                            use_bias=False,
                                            input_shape=input_shape
                                            ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((init_size, init_size, init_channels)))
        channels=init_channels//2
        model.add(tf.keras.layers.Conv2DTranspose(channels, kernel_size,
                                        strides=(1,1),padding='same',
                                        use_bias=False))

        assert model.output_shape==(None, init_size, init_size, 128)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        #double size at each step
        for ii in range(num_upsample-1):
            channels = channels//2
            model.add(tf.keras.layers.Conv2DTranspose(channels, kernel_size,
                                            strides=(2,2), padding='same',
                                            use_bias=False))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())
        # at the final step, collapse to a single channel, apply tanh
        model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size, strides=(2,2),
                                                padding='same', use_bias=False,
                                                activation='tanh'))
        assert model.output_shape == (None, *output_shape, 1)
        return model

    def build_discriminator(self):
        """ Also stolen from the TF DCGAN tutorial"""
        model = tf.keras.Sequential()
        filter_sizes=[64, 128]
        for f in filter_sizes:
            model.add(tf.keras.layers.Conv2D(f, (5,5), strides=(2,2), padding='same'))
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.Dropout(.3))
        # collapse to a single dimension
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model
