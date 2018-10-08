import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt

def create_place_holder(im_h,im_w):
	x = tf.placeholder(shape=[None, n_H0, n_W0, 2],dtype = tf.float32)
	y = tf.palceholder(shape=[None, n_H0, n_W0],dtype = tf.float32)

def manifold_net(input_x):

	x_flatten = tf.contrib.layer.flatten(input_x)
	output_size = np.int(input_x.shape[1] * input_x.shape[2])
	fc1 = tf.tanh(tf.layers.dense(x_flatten,output_size))
	fc2 = tf.tanh(tf.layers.dense(fc1,output_size))
	fcm = tf.reshape(fc2, [tf.shape(input_x)[0], tf.shape(input_x)[1], tf.shape(input_x)[2], 1])

	conv_1 = tf.contrib.layers.convolution2d(fcm, num_outputs=64, kernel_size=5, stride=4, activation_fn=tf.nn.relu)
	conv_2 = tf.contrib.layers.convolution2d(conv_1, num_outputs=64, kernel_size=5, stride=4, activation_fn=tf.nn.relu)
	batch_size = input_x.shape[0]
	deconv_shape = tf.stack([batch_size, x.shape[1], x.shape[2], 1])
	deconv = tf.contrib.layers.conv2d_transpose(conv_2,deconv_shape,kernel_size=7,stride = 4, activation_fn=tf.nn.relu)
	deconv = tf.squeeze(deconv)

	return deconv


def compute_cost(deconv,Y):

	# compute the loss of the label and the inputs
	loss = tf.reduce_mean(tf.square(deconv-Y))
	return loss

def random_mini_batches(x, y, mini_batch_size=64, seed=0):
    """ Shuffles training examples and partitions them into mini-batches
    to speed up the gradient descent
    :param x: input frequency space data
    :param y: input image space data
    :param mini_batch_size: mini-batch size
    :param seed: can be chosen to keep the random choice consistent
    :return: a mini-batch of size mini_batch_size of training examples
    """

    m = x.shape[0]  # number of input images
    mini_batches = []
    np.random.seed(seed)

    # Shuffle (x, y)
    permutation = list(np.random.permutation(m))
    shuffled_X = x[permutation, :]
    shuffled_Y = y[permutation, :]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(
        m / mini_batch_size))  # number of mini batches of size mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:k * mini_batch_size
                                    + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:k * mini_batch_size
                                    + mini_batch_size, :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches
                                  * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches
                                  * mini_batch_size: m, :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
