import os
import glob
import shutil
import pathlib

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from utils.vegetation_model import VegClassModelFCINF
from utils.dataset_loader import get_image_ds, preprocess_image

def load_dataset(dir, ext):

	print('[info] loading dataset...')

	data_root = pathlib.Path(dir)
	image_paths_ = list(data_root.glob('*.'+ext))
	image_paths = [str(path) for path in image_paths_]

	ds, num_images = get_image_ds(dir, 1, ext, shuffle=False)
	image_feed = iter(ds.take(num_images))
	return image_feed, image_paths


def mainfold_reduction(image_dir, model_path, output_dir, manifold):

	n_clusters = 6

	model = VegClassModelFCINF()
	model.load_weights(model_path)

	image_feed, image_paths = load_dataset(image_dir, 'JPG')

	features_arr = []

	print('[info] running autoencoder inference...')
	features_arr = np.array([model(image)[0] for image in image_feed])

	print('[info] embedding features and training k-means...')
	if manifold == 'tsne':
		embedded = TSNE(n_components=2).fit_transform(features_arr)
		kmeans = KMeans(n_clusters=n_clusters).fit(embedded)
		embedded = np.hstack((embedded, kmeans.labels_.reshape(-1, 1)))
	elif manifold == 'umap':
		embedded = umap.UMAP(n_components=2).fit_transform(features_arr)
		kmeans = KMeans(n_clusters=n_clusters).fit(embedded)
		embedded = np.hstack((embedded, kmeans.labels_.reshape(-1, 1)))
	else:
		raise ValueError('manifold must be either tsne or umap.')

	if os.path.isdir(output_dir):
		shutil.rmtree(output_dir)
	os.mkdir(output_dir)

	[os.mkdir(os.path.join(output_dir, str(int(i)))) for i in np.unique(embedded[:, 2])]

	for idx, path in enumerate(image_paths):
		img = cv2.imread(path)
		cv2.imwrite(os.path.join(output_dir, str(int(embedded[idx, 2])), \
								 path.split('/')[-1].split('.')[0]+'.jpg'), img)


	plt.figure()
	plt.title('t-SNE') if manifold == 'tsne' else 'UMAP'
	plt.scatter(embedded[:, 0], embedded[:, 1], c=embedded[:,2], cmap='gist_rainbow')
	plt.colorbar()
	plt.show()

def kmean():




if __name__ == '__main__':

	image_dir = './data/test'
	model_path = './logs/models/model_600.ckpt'
	output_dir = './outputs'

	mainfold_reduction(image_dir, model_path, output_dir, 'umap')
