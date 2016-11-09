from keras.datasets import mnist
from keras.layers import Activation, Convolution2D, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.utils import np_utils

K = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('f') / 255.0  # (60000, 28, 28)
x_test = x_test.astype('f') / 255.0
y_train = np_utils.to_categorical(y_train, K)  # int -> one-of-vector
y_test = np_utils.to_categorical(y_test, K)

model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(28, 28)))  # tensorflow-order!!
model.add(Convolution2D(8, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
model.add(Convolution2D(16, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(K))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=30, nb_epoch=10, validation_data=(x_test, y_test))
