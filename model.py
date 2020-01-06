import tensorflow as tf

class SNConv2DTranspose(tf.keras.layers.Layer):
	def __init__(self, outfilter, 
			kernel_size, 
			stride, 
			name, 
			activation = None, 
			padding = 'SAME'):
		super(SNConv2DTranspose, self).__init__(name = name)
		self.outfilter = outfilter
		self.stride = stride
		if isinstance(kernel_size, int):
			self.kernel = [kernel_size, kernel_size]
		else:
			self.kernel = kernel_size
		self.activation = activation
		self.padding = padding
	def build(self, input_shape):
		self.kernel = self.add_variable("kernel",
										shape = self.kernel + [self.outfilter, int(input_shape[-1])],
										initializer = tf.keras.initializers.GlorotUniform)
		self.bias = self.add_variable("bias",
										shape = [1, self.outfilter],
										initializer = tf.zeros)
		self.u = self.add_variable("u",
									shape = [self.outfilter, 1],
									initializer = tf.random.normal,
									trainable = False)

		self._output_shape = input_shape.as_list()
		self._output_shape[1] *= self.stride
		self._output_shape[2] *= self.stride
		self._output_shape[3] = self.outfilter
		print (self._output_shape)

	def call(self, x, test = False):
		def _power_iteration(w):
			_v = tf.matmul(tf.transpose(w), self.u)
			v = _v / tf.math.l2_normalize(_v)
			_u = tf.matmul(w, v)
			self.u.assign(_u / tf.math.l2_normalize(_u))
			w = w / (tf.matmul(tf.matmul(tf.transpose(self.u), w), v))
			return w

		if not test:
			_kernel = tf.transpose(self.kernel, perm = [0, 1, 3, 2])
			w = tf.transpose(tf.reshape(_kernel, [-1, self.outfilter]))
			w = tf.transpose(_power_iteration(w))
			dims = self.kernel.get_shape().as_list()
			_kernel = tf.reshape(w, [dims[0], dims[1], dims[3], dims[2]])
			self.kernel.assign(tf.transpose(_kernel, perm = [0, 1, 3, 2]))

		self._output_shape[0] = x.get_shape().as_list()[0]
		x = tf.nn.conv2d_transpose(x, self.kernel, self._output_shape, self.stride, self.padding)
		x = x + self.bias
		if callable(self.activation):
			x = self.activation(x)

		return x

class SNConv2D(tf.keras.layers.Layer):
	def __init__(self, outfilter, 
			kernel_size, 
			stride, 
			name, 
			activation = None, 
			padding = 'SAME'):
		super(SNConv2D, self).__init__(name = name)
		self.outfilter = outfilter
		self.stride = stride
		if isinstance(kernel_size, int):
			self.kernel = [kernel_size, kernel_size]
		else:
			self.kernel = kernel_size
		self.activation = activation
		self.padding = padding
	def build(self, input_shape):
		self.kernel = self.add_variable("kernel", 
										shape = self.kernel + [int(input_shape[-1]), self.outfilter],
										initializer = tf.keras.initializers.GlorotUniform)
		self.bias = self.add_variable("bias", 
										shape = [1, self.outfilter],
										initializer = tf.zeros)
		self.u = self.add_variable("u",
									shape = [self.outfilter, 1],
									initializer = tf.random.normal,
									trainable = False)
	def call(self, x, test = False):
		def _power_iteration(w):
			_v = tf.matmul(tf.transpose(w), self.u)
			v = _v / tf.math.l2_normalize(_v)
			_u = tf.matmul(w, v)
			self.u.assign(_u / tf.math.l2_normalize(_u))
			w = w / (tf.matmul(tf.matmul(tf.transpose(self.u), w), v))
			return w

		if not test:
			w = tf.transpose(tf.reshape(self.kernel, [-1, self.outfilter]))
			w = tf.transpose(_power_iteration(w))
			self.kernel.assign(tf.reshape(w, self.kernel.get_shape().as_list()))

		x = tf.nn.conv2d(x, self.kernel, self.stride, self.padding)
		x = x + self.bias
		if callable(self.activation):
			x = self.activation(x)

		return x

class SNDense(tf.keras.layers.Layer):
	def __init__(self, outfilter, 
			name, 
			activation = None):
		super(SNDense, self).__init__(name = name)
		self.outfilter = outfilter
		self.activation = activation
		
	def build(self, input_shape):
		self.kernel = self.add_variable("kernel", 
										shape = [int(input_shape[-1]), self.outfilter],
										initializer = tf.keras.initializers.GlorotUniform)
		self.bias = self.add_variable("bias", 
										shape = [1, self.outfilter],
										initializer = tf.zeros)
		self.u = self.add_variable("u",
									shape = [self.outfilter, 1],
									initializer = tf.random.normal,
									trainable = False)
	def call(self, x, test = False):
		def _power_iteration(w):
			_v = tf.matmul(tf.transpose(w), self.u)
			v = _v / tf.math.l2_normalize(_v)
			_u = tf.matmul(w, v)
			self.u.assign(_u / tf.math.l2_normalize(_u))
			w = w / (tf.matmul(tf.matmul(tf.transpose(self.u), w), v))
			return w

		if not test:
			w = tf.transpose(self.kernel)
			w = tf.transpose(_power_iteration(w))
			self.kernel.assign(w)

		x = tf.matmul(x, self.kernel) + self.bias
		if callable(self.activation):
			x = self.activation(x)

		return x

class Discriminator(tf.keras.Model):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = SNConv2D(64, 5, 2, activation = tf.nn.leaky_relu, name = 'dis_conv1')
		self.conv2 = SNConv2D(128, 5, 2, activation = tf.nn.leaky_relu, name = 'dis_conv2')
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
		self.deconv1 = SNConv2DTranspose(64, 5, 2, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'gen_deconv1')
		self.deconv2 = SNConv2DTranspose(32, 5, 2, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'gen_deconv2')
		self.deconv3 = SNConv2D(1, 3, 1, activation = tf.nn.tanh, padding = 'SAME', name = 'gen_deconv3')

	def call(self, x, test = False):
		x = self.fc1(x)
		x = self.reshape(x)
		x = self.deconv1(x, test = test)
		x = self.deconv2(x, test = test)
		x = self.deconv3(x, test = test)
		return x

