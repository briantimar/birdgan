import numpy as np
from scipy.ndimage import imread

#uniform size for processed images
IMSIZE=(32,32)

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
    return im[lx//2-dx: lx//2 + dx + 1, ly//2 - dy: ly//2 + dy + 1, ...]
