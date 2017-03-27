import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Activation, Dense, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils

M = 10

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('f') / 255.0  # (60000, 28, 28)
x_test = x_test.astype('f') / 255.0
y_train = np_utils.to_categorical(y_train, M)  # int -> one-of-vector
y_test = np_utils.to_categorical(y_test, M)

with tf.Graph().as_default():
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    # model construction
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))  # tensorflow-order!!
    model.add(Conv2D(8, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(M))
    model.add(Activation("softmax"))

    tbd = TensorBoard(log_dir='.log', histogram_freq=1)
    callbacks = [tbd]

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # learning
    model.fit(x_train, y_train, batch_size=30, epochs=10,
              callbacks=callbacks,
              validation_data=(x_test, y_test))
