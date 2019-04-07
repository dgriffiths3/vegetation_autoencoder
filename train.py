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

@tf.function
def train_step(step, model, loss_object, optimizer, image, label, train_loss, train_accuracy):

	with tf.GradientTape() as tape:

		predictions = model(image)
		loss = loss_object(label, predictions)
		#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predictions)

	if step % 100 == 0:
		plot_images(label[0], predictions[0])

	gradients = tape.gradient(loss, model.trainable_variables)

	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	loss = train_loss(loss)
	acc = train_accuracy(label, predictions)
	return loss, acc

@tf.function
def test_step(model, image, label, loss_object, test_loss, test_accuracy):

	predictions = model(image)

	t_loss = loss_object(label, predictions)

	test_loss(t_loss)
	test_accuracy(label, predictions)

def train(image_feed):

	lr = 0.0001
	log_freq = 1

	model = VegClassModel()

	loss_object = tf.keras.losses.MeanSquaredError()
	optimizer = tf.keras.optimizers.Adam(lr=lr)
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
	test_loss = tf.keras.metrics.Mean(name='test_loss')
	test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

	epochs = 5
	step = 0

	next(image_feed)

	print('[info] initialising training...')

	for step, train_images in enumerate(image_feed):

		step += 1
		loss, acc = train_step(step, model, loss_object, optimizer, train_images, train_images, train_loss, train_accuracy)

		if step % 1 == 0:
			print('step: {:}, loss: {:.2f}'.format(step, loss))
		if step % log_freq == 0 or step == 1:
			tf.summary.scalar('loss', loss, step=optimizer.iterations)

	for step, test_images in enumerate(image_feed):

		test_step(model, test_images, test_images, loss_object, test_loss, test_accuracy)

	template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
	print (template.format(epoch+1,
					 train_loss.result(),
					 train_accuracy.result()*100,
					 test_loss.result(),
					 test_accuracy.result()*100))

if __name__ == '__main__':

	log_dir = '.'
	batch_size = 8
	epochs = 1

	print('[info] loading dataset...')
	ds, num_images = get_image_ds('./split_images', batch_size)

	steps_per_epoch=tf.math.ceil(num_images/batch_size).numpy()
	batches=steps_per_epoch*epochs
	image_feed = iter(ds.take(batches+1))

	train_summary_writer = tf.summary.create_file_writer(log_dir+'/logs')

	with train_summary_writer.as_default():
		train(image_feed)
