import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

train, test = chainer.datasets.get_mnist()

use_gpu = True
gpu_device = 2


class Model(chainer.Chain):

    def __init__(self):
        super().__init__(
            conv1=L.Convolution2D(1, 8, 5, stride=2),
            conv2=L.Convolution2D(8, 16, 5, stride=2),
            lin=L.Linear(None, 10),
        )

    def __call__(self, x, t):
        n = x.data.shape[0]
        h = F.reshape(x, (n, 1, 28, 28))
        h = F.elu(self.conv1(h))
        h = F.elu(self.conv2(h))
        h = self.lin(h)
        loss = F.softmax_cross_entropy(h, t)
        acc = F.accuracy(h, t)
        chainer.report({'loss': loss, 'acc': acc}, self)
        return loss


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
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/acc', 'validation/main/acc']))

trainer.run()
