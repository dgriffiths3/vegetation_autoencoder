import tensorflow as tf
import os
import cv2
import random
import pathlib
import matplotlib.pyplot as plt


def split_images(input_dir, output_dir, image_size=[600, 600]):

	data_root = pathlib.Path(input_dir)
	all_image_paths = list(data_root.glob('*'))
	all_image_paths = [str(path) for path in all_image_paths]

	height, width = image_size

	for path in all_image_paths:

		filename = path.split('/')[-1].split('.')[0]
		ext = '.' + path.split('/')[-1].split('.')[1]

		img = cv2.imread(path)

		num_cols = img.shape[1] / width
		num_rows = img.shape[0] / height

		count = 0

		for col in range(int(num_cols)):
			for row in range(int(num_rows)):
				y0 = row * height
				y1 = y0 + height
				x0 = col * width
				x1 = x0 + width
				count += 1
				img_chunk = img[y0:y1, x0:x1]
				cv2.imwrite(os.path.join(output_dir, filename + '_' + str(count) + ext), img_chunk)


def preprocess_image(image):
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, [384, 512])
	image /= 255.0
	return image

def load_and_preprocess_image(path):
	image = tf.io.read_file(path)
	return preprocess_image(image)

def change_range(image):
	return 2*image-1

def get_image_ds(image_dir, batch_size):

	AUTOTUNE = tf.data.experimental.AUTOTUNE

	data_root = pathlib.Path(image_dir)
	all_image_paths = list(data_root.glob('*.JPG'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)

	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

	image_ds = image_ds.shuffle(buffer_size=len(all_image_paths))
	image_ds = image_ds.repeat()
	image_ds = image_ds.batch(batch_size)
	image_ds = image_ds.prefetch(buffer_size=AUTOTUNE)

	image_ds = image_ds.map(change_range)

	return image_ds, len(all_image_paths)

if __name__ == '__main__':

	split_images('./images', './split_images')
