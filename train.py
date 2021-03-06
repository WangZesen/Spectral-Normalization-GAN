from model import *
from utils import *
import tensorflow as tf

def train(args):
	dataset = get_dataset(args)

	generator = Generator()
	discriminator = Discriminator()

	gen_opt = tf.keras.optimizers.Adam(args.learning_rate, beta_1 = 0.5)
	dis_opt = tf.keras.optimizers.Adam(args.learning_rate * 2, beta_1 = 0.5)

	gen_metric = tf.keras.metrics.Mean(name = 'Generator_Loss')
	dis_metric = tf.keras.metrics.Mean(name = 'Discriminator_Loss')

	def train_one_step(real_sample):

		z = tf.random.normal([args.batch_size, args.noise_dim])

		with tf.GradientTape(persistent = True) as tape:
			tape.watch(generator.trainable_variables + discriminator.trainable_variables)
			fake_sample = generator(z)

			all_samples = tf.concat([fake_sample, real_sample], axis = 0)
			all_logit = discriminator(all_samples)
			fake_logit, real_logit = tf.split(all_logit, 2, 0)

			dis_loss = tf.reduce_mean(tf.maximum(1 - real_logit, 0)) + tf.reduce_mean(tf.maximum(1 + fake_logit, 0))
			gen_loss = - tf.reduce_mean(fake_logit)

		gen_gradient = tape.gradient(gen_loss, generator.trainable_variables)
		dis_gradient = tape.gradient(dis_loss, discriminator.trainable_variables)
		gen_opt.apply_gradients(zip(gen_gradient, generator.trainable_variables))
		dis_opt.apply_gradients(zip(dis_gradient, discriminator.trainable_variables))

		del tape

		gen_metric(gen_loss)
		dis_metric(dis_loss)

	def test_step():
		z = tf.random.normal([args.batch_size, args.noise_dim])
		fake_sample = generator(z, test = True)
		return fake_sample

	plot(test_step(), 'samples', 0)
	for i in range(400):

		for batch in dataset:
			train_one_step(batch)

		print (f'Epoch {i + 1}, Gen Loss: {gen_metric.result()}, Dis Loss: {dis_metric.result()}')
		
		if (i + 1) % 2 == 0:
			plot(test_step(), 'samples', i + 1)




