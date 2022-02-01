import functools

import flax.linen
import optax
import tensorflow
import tensorflow_datasets
from flax.training.train_state import TrainState

import jax
import jax.random
from jax import numpy


# dataset
def preprocessing(x, y):
    x = tensorflow.cast(x, tensorflow.float32) / 255.0
    return x, y


ds = tensorflow_datasets.load("mnist", as_supervised=True, download=True)
train_set = ds["train"]
train_set = (
    train_set.shuffle(len(train_set), seed=0, reshuffle_each_iteration=True)
    .batch(32)
    .map(preprocessing)
    .prefetch(1)
)
test_set = ds["test"]
test_set = test_set.batch(32).map(preprocessing).prefetch(1)


# model
class Net(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(128)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(32)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(10)(x)
        x = flax.linen.log_softmax(x)
        return x


model = Net()
params = model.init(jax.random.PRNGKey(42), numpy.ones((1, 28 * 28)))["params"]
optimizer = optax.adam(0.001)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


@functools.partial(jax.jit, static_argnums=(3,))
def step(x, y, state: TrainState, training: bool):
    def loss_fn(params):
        y_pred = model.apply({"params": params}, x)
        y_one_hot = jax.nn.one_hot(y, 10)
        loss = optax.softmax_cross_entropy(y_pred, y_one_hot).mean()
        return loss, y_pred

    x = x.reshape(-1, 28 * 28)
    if training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, y_pred), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
    else:
        loss, y_pred = loss_fn(state.params)
    return loss, y_pred, state


# training
EPOCHS = 10
VIEW_INTERVAL = 200
for epoch in range(EPOCHS):
    running_loss = 0
    for i, (x, y) in enumerate(train_set.as_numpy_iterator()):
        loss, y_pred, state = step(x, y, state, training=True)
        running_loss += loss

        # report loss
        running_loss += loss.item()
        if i % VIEW_INTERVAL == VIEW_INTERVAL - 1:
            print(f"Epoch {epoch+1}, iteration {i+1}; loss: {(running_loss / VIEW_INTERVAL):.3f}")
            running_loss = 0

            # testing
            count_correct = 0
            count_total = 0
            for x, y in test_set.as_numpy_iterator():
                _loss, y_pred, _ = step(x, y, state, training=False)
                count_correct += (numpy.argmax(y_pred, 1) == y).sum()
                count_total += len(y_pred)
            print(f"  Test Acc: {100.0 * count_correct / count_total :.2f}%")
