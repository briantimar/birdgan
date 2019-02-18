# BirdGAN: the GAN for birds

## References
* GANs [for cats][catgan] by AlexiaJM
* [GAN hacks][ganhacks]
* [DCGAN paper][dcgan]
* [Nice visual guide][conv vis] on 'transposed' convolution. Note that the word 'tranpose' here refers to flipping around the conv filter with respect to the input and output tensors. If you consider a conv kernel as mapping patches of pixels to single pixels, then:
  * in the standard conv case, patches are drawn from the larger tensor, and single values placed into the smaller --> downsampling.
  * in the transposed conv, patches are drawn from the smaller tensor, and single values fed into the larger --> upsampling.

  In each case the frames can be zero-padded as necessary.
  One other thing worth noting about the transpose conv: the *stride* has a slightly different meaning in the transpose case than in the standard case. In a standard conv, it controls how far the patch steps across the input when the output pixel is shifted over by one. In the transpose case, the patch always steps by one, while the stride controls the spacing between non-zero-padding pixels in the input. Easiest to just look at the pictures.

* [TF demo][tf DCGAN] of DCGAN training on MNIST

[ganhacks]: https://github.com/soumith/ganhacks
[catgan]: https://github.com/AlexiaJM/Deep-learning-with-cats
[dcgan]: https://arxiv.org/pdf/1511.06434.pdf
[conv vis]: https://github.com/vdumoulin/conv_arithmetic
[tf DCGAN]: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb
