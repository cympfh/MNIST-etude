#
# Achieve Acc=91%
#
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train)  # <tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x..>

with tf.Session() as sess:

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    y = tf.matmul(x, W) + b

    e = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))  # loss

    acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)),
        tf.float32))

    sess.run(tf.initialize_all_variables())
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(e)

    for i in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        a1 = acc.eval(feed_dict={x: batch[0], y_: batch[1]})  # acc on training
        a2 = acc.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})  # acc on test
        print("Acc: {}, {}".format(a1, a2))
