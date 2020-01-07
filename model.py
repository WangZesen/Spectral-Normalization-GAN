import tensorflow as tf
from SNLayer import *

class Discriminator(tf.keras.Model):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = SNConv2D(64, 7, 2, activation = tf.nn.leaky_relu, name = 'dis_conv1')
		self.conv2 = SNConv2D(128, 7, 2, activation = tf.nn.leaky_relu, name = 'dis_conv2')
		self.flatten = tf.keras.layers.Flatten()
		self.fc1 = SNDense(1, name = 'dis_fc1')

	def call(self, x, test = False):
		x = self.conv1(x, test = test)
		x = self.conv2(x, test = test)
		x = self.flatten(x)
		x = self.fc1(x, test = test)
		# x = self.fc2(x)
		return x

class Generator(tf.keras.Model):
	def __init__(self):
		super(Generator, self).__init__()
		def _reshape_func(x):
			dims = x.get_shape().as_list()
			return tf.reshape(x, [dims[0], 7, 7, 128])
		self.fc1 = SNDense(7 * 7 * 128, name = 'gen_fc1')
		self.reshape = _reshape_func
		self.deconv1 = SNConv2DTranspose(128, 7, 2, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'gen_deconv1')
		self.deconv2 = SNConv2DTranspose(128, 7, 2, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'gen_deconv2')
		self.deconv3 = SNConv2DTranspose(64, 7, 1, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'gen_deconv3')
		self.deconv4 = SNConv2D(1, 3, 1, activation = tf.nn.tanh, padding = 'SAME', name = 'gen_deconv4')

	def call(self, x, test = False):
		x = self.fc1(x)
		x = self.reshape(x)
		x = self.deconv1(x, test = test)
		x = self.deconv2(x, test = test)
		x = self.deconv3(x, test = test)
		x = self.deconv4(x, test = test)
		return x

