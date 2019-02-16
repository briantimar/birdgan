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
    return (.5 *im + .5) * 255

def preprocess_image(fname, size=(64,64), normalize=True):
    """ fname = filename of bird image.
        returns: float tensor of the requested spatial size."""
    str_tensor = tf.read_file(fname)
    #load the image from jpg
    im = tf.image.decode_jpeg(str_tensor,channels=3)
    #resize as requested
    im = tf.image.resize_images(im,size=size)
    #normalize
    if normalize:
        im =apply_normalization(im)
    return im

def get_all_image_filenames(target="./data/birds_all/"):
    """ returns list of all bird filenames"""
    import glob
    return glob.glob(target+"*")

def make_string_dataset(fnames):
    """ Make a tf string dataset using the images specified in list of filename strings
    fnames"""
    N=len(fnames)
    print("building dataset from %d images" % N)
    filenames = tf.constant(fnames)
    return tf.data.Dataset.from_tensor_slices(filenames)

def make_dataset(fnames, size=(64,64), normalize=True):
    """ Construct tf dataset holding preprocessed images.
        size=spatial extent of images in the dataset."""
    preproc = lambda f: preprocess_image(f, size=size, normalize=normalize)
    strd = make_string_dataset(fnames)
    #apply preproc
    return strd.map(preprocessor)
