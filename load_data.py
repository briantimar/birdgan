import numpy as np
from scipy.ndimage import imread

#uniform size for processed images
IMSIZE=(64, 64)

#downsampling factor along each spatial axis
DS = 5

def get_raw(fname):
    """ Load raw image with specified filepath as np array"""
    return imread(fname)

def downsample(im):
    return im[::DS, ::DS,...]

def crop(im):
    """center crop to IMSIZE"""
    lx, ly, _ = im.shape
    dx,dy=IMSIZE[0]//2, IMSIZE[1]//2
    return im[lx//2-dx: lx//2 + dx, ly//2 - dy: ly//2 + dy, ...]

def resize(im):
    """ down sample and center crop"""
    return crop(downsample(im))


def preprocess(im):
    """ apply all desired preprocessing to numpy array representing raw image"""
    return resize(im)
