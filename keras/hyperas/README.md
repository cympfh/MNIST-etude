
# Keras+Hyperas

## results

### conv.py

#### stdout:

```
Evalutation of best performing model:
   ['loss', 'acc'] [0.21304182317852974, 0.9473]
Best performing model chosen hyper-parameters:
   {'Conv2D': 0, 'Conv2D_1': 2, 'Dropout': 0.5350807190884803, 'Dropout_1': 0.9203644803497606, 'batch_size': 1}
```

#### best model:

```python
M = 10
model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(Conv2D(8, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(0.5350807190884803))
model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(0.9203644803497606))
model.add(Flatten())
model.add(Dense(M, activation='softmax'))
```
