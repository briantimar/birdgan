import tensorflow as tf
import numpy as np
from models import DCGAN, make_losses
from preprocess_data import make_dataset, get_all_image_filenames
import time
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

num_examples = 4**2


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

z = tf.random_normal(noise_shape)

losses = make_losses(z, tr_feed, generator, discriminator)

sampling_noise_seed =np.random.normal(size=(num_examples, noise_dim))
noise_feed = tf.placeholder(dtype=tf.float32, shape=(num_examples, noise_dim))
#holds images sampled from the generator during training
sampled_images = generator(noise_feed,training=False)

gen_optim = tf.train.AdamOptimizer(lr)
disc_optim = tf.train.AdamOptimizer(lr)

gen_tr_op = gen_optim.minimize(losses['generator'], var_list=generator.variables)
disc_tr_op = disc_optim.minimize(losses['discriminator'], var_list = discriminator.variables)

batches_per_epoch = N//batch_size

#number of models to save
num_save = 10
save_step = batches_per_epoch * epochs // num_save
summary_step=10
batch = 0

fw = tf.summary.FileWriter("logs", graph=tf.get_default_graph())
tf.summary.scalar("gen_loss", losses['generator'])
tf.summary.scalar("disc_loss", losses['discriminator'])
summary_all = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=num_save)

def plot_samples(ep, samples):
    from preprocess_data import undo_normalization
    samples = undo_normalization(samples)
    fig = plt.figure(figsize=(4,4))
    for i in range(samples.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(samples[i, ...])
        plt.axis('off')
    plt.savefig("images/samples_at_ep_%d.png"%ep)

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
            __,__, summary_ = sess.run([disc_tr_op, gen_tr_op, summary_all])

            if batch % batches_per_epoch==0:
                ep=batch//epochs
                print("time for epoch {1}: {0:.2e} sec".format(time.time()-t, ep))
                sampled_images_ = sess.run(sampled_images,
                                            feed_dict={noise_feed:sampling_noise_seed})
                plot_samples(ep, sampled_images_)

            if batch % save_step ==0:
                saver.save(sess, "models/saved_model", global_step=batch//save_step)
            if batch % summary_step==0:
                fw.add_summary(summary_, global_step=batch//summary_step)
            batch +=1

        except tf.errors.OutOfRangeError:
            break
