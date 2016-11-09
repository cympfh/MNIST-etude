import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class Model(chainer.Chain):

    def __init__(self):
        super().__init__(lin=L.Linear(28 * 28, 10))

    def __call__(self, x, t):
        y = self.lin(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        chainer.report({'loss': loss, 'acc': acc}, self)
        return loss


model = Model()
opt = chainer.optimizers.SGD()
opt.setup(model)

bs = 100
train, test = chainer.datasets.get_mnist()
train_iter = chainer.iterators.SerialIterator(train, bs)
test_iter = chainer.iterators.SerialIterator(test, bs, repeat=False)

updater = chainer.training.StandardUpdater(train_iter, opt)
trainer = chainer.training.Trainer(updater, (10, 'epoch'))
trainer.extend(extensions.Evaluator(test_iter, model), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/acc', 'validation/main/acc']))

trainer.run()
