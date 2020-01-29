import thinc.api as api

import cowsay
import ml_datasets

# configure
config = api.Config()
config.from_disk('./linear.cfg')
loaded_config = api.registry.make_from_config(config)
batch_size = loaded_config['training']['batch_size']
n_iter = loaded_config['training']['n_iter']

# dataset
(train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
cowsay.cow(f"Training size={len(train_X)}, dev size={len(dev_X)}")


# model
model = api.Softmax()
model.initialize(X=train_X, Y=train_Y)
cowsay.cow(f"Initialized model with input dimension "
           f"nI={model.get_dim('nI')} and output dimension nO={model.get_dim('nO')}")

api.fix_random_seed(0)
optimizer = loaded_config['optimizer']
print("Training")
for _ in range(n_iter):
    for X, Y in model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True):
        Yh, backprop = model.begin_update(X)
        backprop(Yh - Y)
        model.finish_update(optimizer)

print("Testing")
n_correct = 0
n_total = 0
for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y, shuffle=True):
    Yh = model.predict(X)
    n_correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
    n_total += Yh.shape[0]
cowsay.cow(f"Acc = {n_correct / n_total:.3f}")
