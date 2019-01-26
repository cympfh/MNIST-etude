#
# Acc=95%
#
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')


def batch_normalization(shape, input):
    """
    thanks to http://qiita.com/sergeant-wizard/items/052c98c6e712a4a8df6a
    aka. kopipe
    """
    eps = 1e-5
    gamma = tf.Variable(tf.truncated_normal(shape))
    beta = tf.Variable(tf.truncated_normal(shape))
    mean, var = tf.nn.moments(input, [0])
    return gamma * (input - mean) / tf.sqrt(var + eps) + beta


with tf.Session() as sess:

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])  # width, height, channel

    with tf.device('/gpu:0'):

        with tf.name_scope('conv1'):
            w = weight_variable([5, 5, 1, 8])
            b = weight_variable([8])
            h = tf.nn.relu(conv2d(x_image, w) + b)
            h = max_pool_2x2(h)

        h = batch_normalization([14, 14, 8], h)

        with tf.name_scope('conv2'):
            w = weight_variable([5, 5, 8, 16])
            b = weight_variable([16])
            h = tf.nn.relu(conv2d(h, w) + b)
            h = max_pool_2x2(h)

        k = 16 * 7 * 7
        h = tf.reshape(h, [-1, k])

        with tf.name_scope('out_linear'):
            w = tf.Variable(tf.zeros([k, 10]))
            b = tf.Variable(tf.zeros([10]))
            h = tf.matmul(h, w) + b

    y = h

    e = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))  # loss

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
