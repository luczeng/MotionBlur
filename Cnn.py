from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class MovingBlurCnn:
	@staticmethod
	def build(width, height, depth, weightsPath=None):
			#weightsPath : to use pre-trained model

		# initialize the model
		model = Sequential()

		#  Layers
		model.add(Convolution2D(10,11,11,border_mode ="same",input_shape=(width, height, depth)))
		model.add(Activation("relu"))
		model.add(Convolution2D(10,11,11,border_mode ="same",input_shape=(width, height, depth)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

		model.add(Convolution2D(15,9,9,border_mode="same"))
		model.add(Activation("relu"))
		model.add(Convolution2D(15,9,9,border_mode="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

		model.add(Convolution2D(15,5,5,border_mode="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

		model.add(Convolution2D(20,5,5,border_mode="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


		model.add(Flatten())

		model.add(Dense(1))

		return model
