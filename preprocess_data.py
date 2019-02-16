""" Perform all preprocessing for the bird data.
The bird dataset is composed of (Lx, Ly, 3) RGB arrays, taking values in
[0, 255]. The horizontal and vertical dimensions Lx, Ly vary somewhat from
image to image, typically a few hundred pixels each.

Preprocessing sequence:
    - reshape images to fixed square size
    - apply constant rescaling op to place pixel values in [-1,1]"""

import tensorflow as tf

def apply_normalization(im):
    """Normalize image tensor to take values in [-1,1]"""
    return 2 * (im/255.0 - .5)

def undo_normalization(im):
    """ take a [-1,1] normalized image to tensor to integer [0,255] range tensor"""
    return tf.cast( .5 * (im +1) * 255, tf.int32 )

def preprocess_image(fname, size=(64,64), normalize=True):
    """ fname = filename of bird image.
        returns: float tensor of the requested spatial size."""
    #load the image from jpg
    im = tf.image.decode_jpeg(fname)
    #resize as requested
    im = tf.image.resize_image(im)
    #normalize
    if normalize:
        im =apply_normalization(im)
    return im
