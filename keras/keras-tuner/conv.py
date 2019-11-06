import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D
from tensorflow.keras.models import Sequential

from keras.utils import np_utils
from kerastuner.tuners import RandomSearch

assert tensorflow.__version__ >= '2.0.0'


def data():
    M = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('f') / 255.0  # (60000, 28, 28)
    x_test = x_test.astype('f') / 255.0
    y_train = np_utils.to_categorical(y_train, M)  # int -> one-of-vector
    y_test = np_utils.to_categorical(y_test, M)
    return x_train, y_train, x_test, y_test


def build_model(hp):
    M = 10
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Conv2D(hp.Int('kernel_1', 8, 32, step=8), (5, 5),
                     strides=(2, 2), activation='relu'))
    model.add(Dropout(hp.Float('ratio_1', min_value=0, max_value=0.3)))
    model.add(Conv2D(hp.Int('kernel_2', 16, 64, step=16), (5, 5),
                     strides=(2, 2), activation='relu'))
    model.add(Dropout(hp.Float('ratio_2', min_value=0.0, max_value=0.3)))
    model.add(Flatten())
    model.add(Dense(M, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    X, Y, X_test, Y_test = data()
    tuner = RandomSearch(build_model, objective='val_accuracy',
                         max_trials=10,
                         directory='out', project_name='conv')
    tuner.search(X, Y, epochs=5, validation_data=(X_test, Y_test))
    tuner.results_summary()
