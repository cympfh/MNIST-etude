# MNIST-etude

This project is a collection of MNIST classification.

諸フレームワークに触れることを目的にします.

More examples of variety frameworks and classification methods, rather than higher performance (accuracy).

`(semisup)` stands for "semi-supervised learning."

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
    - [x] vat `(semisup)`
        - Acc 93.3% / 10 epoch, 500 labels
- keras/
    - [x] linear
        - Acc 91.66% / 10 epoch
    - [x] conv
        - Acc 97.42% / 10 epoch
    - [x] rnn
        - Acc 93.66% / 10 epoch
    - [x] Learning by Association `(semisup)`
        - Acc 96.00% / 100 labels
        - Acc 90.30% / 10 labels
        - see [cympfh/learning-by-association-MNIST](https://github.com/cympfh/learning-by-association-MNIST)
- pytorch/
    - [x] linear
        - Acc 91.65% / 10 epoch
    - [x] conv
        - Acc 97.74% / 10 epoch
    - [ ] rnn

# Rule

- Optimizers: `SGD`
- Dataset: `MNIST`
    - 60k items for training
    - 10k items for testing

## Networks

Very simple architectures are adopted.

### linear: has only 1 linear (or dense) layer

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

### rnn

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

