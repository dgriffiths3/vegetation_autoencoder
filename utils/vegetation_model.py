import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, Reshape
from tensorflow.keras import Model


class VegClassModel(Model):
	def __init__(self):
		# Model architecture definition
		super(VegClassModel, self).__init__()

		self.activation = tf.nn.relu
		self.kernel_initializer = 'random_normal'

		# Encoder
		self.conv1 = Conv2D(filters=32,
							kernel_size=3,
							activation=self.activation,
							padding='same',
							kernel_initializer=self.kernel_initializer)
		self.mp1 = MaxPool2D(pool_size=(2, 2),
							 padding='same')

		self.conv2 = Conv2D(filters=64,
							kernel_size=3,
							activation=self.activation,
							padding='same',
							kernel_initializer=self.kernel_initializer)
		self.mp2 = MaxPool2D(pool_size=(2, 2),
							 	 padding='same')

		self.conv3 = Conv2D(filters=128,
							kernel_size=3,
							activation=self.activation,
							padding='same',
							kernel_initializer=self.kernel_initializer)
		self.encoded = MaxPool2D(pool_size=(2, 2),
							 	 padding='same')

		# Decoder

		self.tconv1 = Conv2DTranspose(filters=128,
									  kernel_size=3,
									  activation=self.activation,
									  padding='same',
									  kernel_initializer=self.kernel_initializer)

		self.up1 = UpSampling2D(size=(2,2),
								interpolation='nearest')

		self.tconv2 = Conv2DTranspose(filters=64,
									  kernel_size=3,
									  activation=self.activation,
									  padding='same',
									  kernel_initializer=self.kernel_initializer)

		self.up2 = UpSampling2D(size=(2,2),
								interpolation='nearest')

		self.tconv3 = Conv2DTranspose(filters=32,
									  kernel_size=3,
									  activation=self.activation,
									  padding='same',
									  kernel_initializer=self.kernel_initializer)

		self.up3 = UpSampling2D(size=(2,2),
								interpolation='nearest')

		self.decoded = Conv2D(filters=3,
							  kernel_size=(3, 3),
							  activation='sigmoid',
							  padding='same',
							  kernel_initializer=self.kernel_initializer)


	def call(self, input):

		net = self.conv1(input)
		net = self.mp1(net)
		net = self.conv2(net)
		net = self.mp2(net)
		net = self.conv3(net)
		net = self.encoded(net)

		net = self.tconv1(net)
		net = self.up1(net)
		net = self.tconv2(net)
		net = self.up2(net)
		net = self.tconv3(net)
		net = self.up3(net)

		logits = self.decoded(net)

		return logits


class VegClassModelFC(Model):
	def __init__(self):
		# Model architecture definition
		super(VegClassModelFC, self).__init__()

		self.activation = tf.nn.relu
		self.kernel_initializer = 'random_normal'

		# Encoder
		self.conv1 = Conv2D(filters=128,
							kernel_size=3,
							activation=self.activation,
							padding='same',
							kernel_initializer=self.kernel_initializer)
		self.mp1 = MaxPool2D(pool_size=(2, 2),
							 padding='same')

		self.conv2 = Conv2D(filters=64,
							kernel_size=3,
							activation=self.activation,
							padding='same',
							kernel_initializer=self.kernel_initializer)
		self.mp2 = MaxPool2D(pool_size=(2, 2),
							 	 padding='same')

		self.conv3 = Conv2D(filters=32,
							kernel_size=3,
							activation=self.activation,
							padding='same',
							kernel_initializer=self.kernel_initializer)
		self.encoded = MaxPool2D(pool_size=(2, 2),
							 	 padding='same')
		# Image is now size 64x64 = 4096
		self.flat = Flatten() # 4096 x 128

		self.fc1 = tf.keras.layers.Dense(units=8192, activation=self.activation)
		self.fc2 = tf.keras.layers.Dense(units=4096, activation=self.activation)

		# Decoder
		self.fc3 = tf.keras.layers.Dense(units=8192, activation=self.activation)

		self.square = Reshape((64, 64, -1)) # Calculate this programmatically

		self.tconv1 = Conv2DTranspose(filters=32,
									  kernel_size=3,
									  activation=self.activation,
									  padding='same',
									  kernel_initializer=self.kernel_initializer)

		self.up1 = UpSampling2D(size=(2,2),
								interpolation='nearest')

		self.tconv2 = Conv2DTranspose(filters=64,
									  kernel_size=3,
									  activation=self.activation,
									  padding='same',
									  kernel_initializer=self.kernel_initializer)

		self.up2 = UpSampling2D(size=(2,2),
								interpolation='nearest')

		self.tconv3 = Conv2DTranspose(filters=128,
									  kernel_size=3,
									  activation=self.activation,
									  padding='same',
									  kernel_initializer=self.kernel_initializer)

		self.up3 = UpSampling2D(size=(2,2),
								interpolation='nearest')

		self.decoded = Conv2D(filters=3,
							  kernel_size=(3, 3),
							  activation='sigmoid',
							  padding='same',
							  kernel_initializer=self.kernel_initializer)

	def call(self, input):

		net = self.conv1(input)
		net = self.mp1(net)
		net = self.conv2(net)
		net = self.mp2(net)
		net = self.conv3(net)
		net = self.encoded(net)

		net = self.flat(net)
		net = self.fc1(net)
		net = self.fc2(net)
		net = self.fc3(net)
		net = self.square(net)

		net = self.tconv1(net)
		net = self.up1(net)
		net = self.tconv2(net)
		net = self.up2(net)
		net = self.tconv3(net)
		net = self.up3(net)

		logits = self.decoded(net)

		return logits
