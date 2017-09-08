# MNIST-etude

単に諸フレームワークに触れることを目的とします

- tensorflow/
    - [x] linear
        - Acc 91%
    - [ ] conv
        - 書いてるけど下のネットワークとは全然違うし精度でないので何か間違ってる
- chainer/
    - [x] linear
        - Acc 90.48% / 10 epoch
    - [x] conv
        - Acc 97.78% / 10 epoch
    - [x] vat
        - Acc 93.3% / 10 epoch
- keras/
    - [x] linear
        - Acc 91.66% / 10 epoch
    - [x] conv
        - Acc 97.42% / 10 epoch
    - [x] rnn
        - Acc 93.66% / 10 epoch
- pytorch/
    - [x] conv
        - Acc 98.44% / 2 epoch

# Setup

## Optimizers

SGD

# Dataset

## full supervised

- 60k items for training
- 10k items for validation (=test)

## semi supervised

- 60k items for training
    - 500 items labeled
    - rest (59,500 items) are unlabeld
- 10k items for validation (=test)

## Networks

### linear

Supervised.
All images be flatten to vectors.

```
28x28 (raw Image)
== 784 (Flatten)
-> 10 (Linear)
-> 10 (Softmax)
```

### conv

Supervised.
A simple CNN.

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

### rnn

Supervised.
All images be flatten to sequences.

```
28x28
== 1x784 (Reshape)
-> 20 (LSTM)
-> 10 (Linear)
-> 10 (Softmax)
```

### VAT (= Convolution + Virtual Adversarial Training)

Semi-supervised.
The labeled items are learned with the simple CNN (previous).
The all (labeled and unlabeld) are used in VAT.

