import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

train, test = chainer.datasets.get_mnist()

N = len(train)
for i in range(500, N):
    train._datasets[1][i] = -1  # unlabeled

use_gpu = True
gpu_device = 3

if use_gpu:
    xp = chainer.cuda.cupy
else:
    import numpy
    xp = numpy


class Model(chainer.Chain):

    def __init__(self):
        super().__init__(
            conv1=L.Convolution2D(1, 8, 5, stride=2),
            conv2=L.Convolution2D(8, 16, 5, stride=2),
            lin=L.Linear(None, 10),
        )

    def forward(self, x):
        n = x.data.shape[0]
        h = F.reshape(x, (n, 1, 28, 28))
        h = F.elu(self.conv1(h))
        h = F.elu(self.conv2(h))
        h = self.lin(h)
        return h

    def to_unit(self, v):
        eps = 1e-6
        v /= xp.abs(v).max()
        norm = xp.linalg.norm(v, axis=1) + eps
        v /= norm.reshape(-1, 1)
        return v

    def vat_loss(self, x, h):
        eps = 1e-6
        xi = 10.0
        phi = 1.0

        y = F.softmax(h) + eps
        d = chainer.Variable(self.to_unit(xp.random.randn(*x.data.shape).astype('f')))

        h2 = self.forward(x.data + xi * d)
        y2 = F.softmax(h2) + eps

        kl_d = F.sum(y.data * (F.log(y.data) - F.log(y2)))
        kl_d.backward()

        d = self.to_unit(d.grad)
        h2 = self.forward(x + phi * d)
        y2 = F.softmax(h2) + eps
        kl_d = F.sum(y.data * (F.log(y.data) - F.log(y2))) / x.data.shape[0]

        return kl_d

    def __call__(self, x, t):
        h = self.forward(x)
        lds = self.vat_loss(x, h)
        loss = F.softmax_cross_entropy(h, t)
        acc = F.accuracy(h, t, ignore_label=-1)
        chainer.report({'loss': loss, 'lds': lds, 'acc': acc}, self)
        return loss + lds


model = Model()
if use_gpu:
    chainer.cuda.get_device(gpu_device).use()
    model.to_gpu()

opt = chainer.optimizers.SGD()
opt.setup(model)

bs = 20
train_iter = chainer.iterators.SerialIterator(train, bs)
test_iter = chainer.iterators.SerialIterator(test, bs, repeat=False)
updater = chainer.training.StandardUpdater(train_iter, opt, device=gpu_device)
trainer = chainer.training.Trainer(updater, (10, 'epoch'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_device), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport([
    'epoch', 'iteration',
    'main/lds', 'main/loss', 'main/acc',
    'validation/main/acc']))

trainer.run()
