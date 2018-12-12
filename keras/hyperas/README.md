# Keras+Hyperas

## conv.py

Acc 98.25% achieved.

### stdout:

```
Evalutation of best performing model:
   ['loss', 'acc'] [0.05856410140097141, 0.9825]
Best performing model chosen hyper-parameters:
   {'Conv2D': 2, 'Conv2D_1': 2, 'Dropout': 0.005547479085219964, 'Dropout_1': 0.1075927951318913}
```

### best model:

```python
M = 10
model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(0.005547479085219964))
model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(0.1075927951318913))
model.add(Flatten())
model.add(Dense(M, activation='softmax'))
```
