# MNIST-etude

単に諸フレームワークに触れることを目的とします

- [x] tensorflow/linear
    - Acc 91%
- [ ] tensorflow/conv
    - 書いてるけど下のネットワークとは全然違うし精度でないので何か間違ってる
- [x] chainer/linear
    - Acc 90.48% / 10 epoch
- [x] chainer/conv
    - Acc 97.78% / 10 epoch
- [x] keras/linear
    - Acc 91.66% / 10 epoch
- [x] keras/conv
    - Acc 97.42% / 10 epoch

# Setup

## Optimizers

SGD

## Networks

### linear

```
28x28 (raw Image)
== 784 (Flatten)
-> 10 (Linear)
-> 10 (Softmax)
```

### conv

```
28x28
== 1x28x28 (Resize)
-> 8x12x12 (Convolution(kernel=5, stride=2))
-> _ (elu)
-> 16x4x4 (Convolution(kernel=5, stride=2))
-> _ (elu)
== 256 (Flatten)
-> 10 (Linear)
-> 10 (Softmax)
```
