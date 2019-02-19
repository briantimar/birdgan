import tensorflow as tf
import numpy as np
from models import DCGAN, make_losses
from preprocess_data import make_dataset, get_all_image_filenames
import time

fnames = get_all_image_filenames()
N=len(fnames)
print("%d images found" % N)

epochs=50
batch_size=128
lr=1e-4

image_set = make_dataset(fnames)
tr_set = image_set.shuffle(N).repeat(epochs).batch(batch_size)
tr_feed = tr_set.make_one_shot_iterator().get_next()

noise_dim=100

noise_shape=(batch_size, noise_dim)

birdgan = DCGAN()
generator = birdgan.build_generator(input_shape=(noise_dim,))
discriminator = birdgan.build_discriminator()

z = tf.random_uniform(noise_shape)


losses = make_losses(z, tr_feed, generator, discriminator)

gen_optim = tf.train.AdamOptimizer(lr)
disc_optim = tf.train.AdamOptimizer(lr)

gen_tr_op = gen_optim.minimize(losses['generator'], var_list=generator.variables)
disc_tr_op = disc_optim.minimize(losses['discriminator'], var_list = discriminator.variables)

Nbatch = (N//batch_size)*epochs

step=50

batch = 0

#important
#if you don't include the 'allow growth' setting, tf will throw a mysterious
# CUDNN error
# see here: https://github.com/tensorflow/tensorflow/issues/6698#issuecomment-297179317

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    while True:
        try:
            t=time.time()
            __,__, losses_ = sess.run([disc_tr_op, gen_tr_op, losses])
            print("time for batch: {0:.2e} sec".format(time.time()-t))
            batch +=1
        except tf.errors.OutOfRangeError:
            break
