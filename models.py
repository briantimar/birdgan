""" tf GAN models """
import tensorflow as tf

#conventions for true and false labels
def get_true_labels(shape):
    """ Labels for real data"""
    return tf.ones(shape, dtype=tf.int32)
def get_fake_labels(shape):
    """ Labels for fake data"""
    return tf.zeros(shape, dtype=tf.int32)

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

    def build_discriminator(self, filter_sizes=[64,128]):
        """ Also stolen from the TF DCGAN tutorial"""
        model = tf.keras.Sequential()

        for f in filter_sizes:
            model.add(tf.keras.layers.Conv2D(f, (5,5), strides=(2,2), padding='same'))
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.Dropout(.3))
        # collapse to a single dimension
        model.add(tf.keras.layers.Flatten())
        #the final layer is a single logit
        model.add(tf.keras.layers.Dense(1))

        return model

def make_gen_loss(gen_logits):
    """ Given logits of the discriminator when evaluated on generator samples,
    return the loss function for the generator.

    Assuming the discriminator outputs a single real value (ie not a )"""

    target_labels = make_true_labels(tf.shape(gen_logits))
    return tf.losses.sigmoid_cross_entropy(target_labels, gen_logits)

def make_discriminator_loss(gen_logits, real_logits):
    """
         loss fn for the discriminator to minimize. """

    gen_labels = make_false_labels(tf.shape(gen_logits))
    real_labels = make_true_labels(tf.shape(real_logits))

    gen_loss = tf.losses.sigmoid_cross_entropy(gen_labels, gen_logits)
    real_loss = tf.losses.sigmoid_cross_entropy(real_labels, real_logits)
    return real_loss + gen_loss

def make_losses(noise_source, real_data, generator, discriminator):

    gen_outputs = generator(noise_source)
    gen_logits = discriminator(gen_outputs)
    real_logits = discriminator(real_data)

    return dict(generator=make_gen_loss(gen_logits),
                discriminator=make_discriminator_loss(gen_logits, real_logits))
