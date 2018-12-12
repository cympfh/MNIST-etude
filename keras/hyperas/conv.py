import numpy

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    M = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('f') / 255.0  # (60000, 28, 28)
    x_test = x_test.astype('f') / 255.0
    y_train = np_utils.to_categorical(y_train, M)  # int -> one-of-vector
    y_test = np_utils.to_categorical(y_test, M)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train):

    M = 10
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Conv2D({{choice([8, 16, 32])}}, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Conv2D({{choice([16, 32, 64])}}, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Flatten())
    model.add(Dense(M, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    result = model.fit(x_train, y_train,
                       batch_size=30,
                       epochs=10,
                       validation_split=0.1)
    losses = result.history['val_loss']
    return {'loss': numpy.amin(losses), 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
    _, _, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print("  ",
          best_model.metrics_names,
          best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print("  ", best_run)
