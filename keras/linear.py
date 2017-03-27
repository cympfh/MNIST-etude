from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils

K = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('f') / 255.0
x_test = x_test.astype('f') / 255.0
y_train = np_utils.to_categorical(y_train, K)  # int -> one-of-vector
y_test = np_utils.to_categorical(y_test, K)

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(K, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=30, epochs=10, validation_data=(x_test, y_test))
