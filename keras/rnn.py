import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.datasets import mnist
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential


seq_len = 28
input_dim = 28
hidden_dim = 128
classes_num = 10


# dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('f') / 255.0  # (60000, 28, 28)
x_test = x_test.astype('f') / 255.0

with tf.Graph().as_default():
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    # model construction
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=(seq_len, input_dim)))
    model.add(Dense(classes_num))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # learning
    callbacks = []
    # from keras.callbacks import TensorBoard
    # tbd = TensorBoard(log_dir='.log', histogram_freq=1)
    # callbacks = [tbd]
    model.fit(x_train, y_train, batch_size=30, epochs=10,
              callbacks=callbacks,
              validation_data=(x_test, y_test))
