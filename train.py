import pandas as pd
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.vegetation_model import VegClassModel
from utils.dataset_loader import get_image_ds

def plot_images(label, prediction):
	plt.figure()
	plt.subplot(121)
	plt.imshow(label)
	plt.title('label')
	plt.subplot(122)
	plt.imshow(prediction)
	plt.title('generated')
	plt.show()


def load_dataset(dir, epochs, batch_size):

	print('[info] loading dataset...')
	ds, num_images = get_image_ds(dir, batch_size)
	steps_per_epoch=tf.math.ceil(num_images/batch_size).numpy()
	batches = steps_per_epoch*epochs
	image_feed = iter(ds.take(batches+1))
	return image_feed


@tf.function
def train_step(step, model, loss_object, optimizer, image, label, train_loss):

	with tf.GradientTape() as tape:

		predictions = model(image)
		loss = loss_object(label, predictions)
		# Alternative loss function
		#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predictions)

	#if step % 100 == 0:
	#	plot_images(label[0], predictions[0])

	gradients = tape.gradient(loss, model.trainable_variables)

	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	loss = train_loss(loss)

	return loss


def train(image_dir):

	batch_size = 8
	epochs = 1
	lr = 0.0001
	log_freq = 1
	save_freq = 1000

	model = VegClassModel()
	image_feed = load_dataset(image_dir, epochs, batch_size)

	loss_object = tf.keras.losses.MeanSquaredError()
	optimizer = tf.keras.optimizers.Adam(lr=lr)
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	test_loss = tf.keras.metrics.Mean(name='test_loss')

	epochs = 5
	step = 0

	next(image_feed)

	print('[info] initialising training...')

	for step, train_images in enumerate(image_feed):

		step += 1
		loss = train_step(step, model, loss_object, optimizer, train_images, train_images, train_loss)

		if step % 1 == 0:
			print('step: {:}, loss: {:.2f}'.format(step, loss))
		if step % log_freq == 0 or step == 1:
			tf.summary.scalar('loss', loss, step=optimizer.iterations)
		if step % save_freq == 0:
			tf.saved_model.save(model, os.path.join(log_dir, 'models', 'model_'+str(step)+'.ckpt'))


if __name__ == '__main__':

	image_dir = './split_images'
	log_dir = '.'

	train_summary_writer = tf.summary.create_file_writer(log_dir+'/logs')

	with train_summary_writer.as_default():
		train(image_dir)
