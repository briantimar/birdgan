## A few practical notes

### Train time
I'm using an RTX2070 GPU for training. Currently, when using 64x64 inputs this takes
.2 sec to process a single batch of 128 images. The full dataset holds 6000 images,
so that's about 12sec / epoch. Training on CPU takes about 8 times as long.

### GPU memory issues?
The first time I tried to train a BirdGAN, tensorflow crashed with the following error:
```
2019-02-19 00:23:57.686368: E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
2019-02-19 00:23:57.688629: E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
2019-02-19 00:23:57.688648: F tensorflow/core/kernels/conv_grad_input_ops.cc:985] Check failed: stream->parent()->GetConvolveBackwardDataAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(stream->parent()), &algorithms)
```

A bit of googling led me to [this](https://github.com/tensorflow/tensorflow/issues/6698#issuecomment-297179317) comment on a similar bug report -- this seems to resolve the issue in my case. Apparently, TF tries to grab all available GPU memory, all at once, and this leads to problems. See also [TF guide to GPU training](https://www.tensorflow.org/guide/using_gpu).
