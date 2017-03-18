import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD


#Input Definitions
img_rows, img_cols, img_channels = 224, 224, 125
learning_rate = 0.001
batch_size = 256
model = Sequential()

#Convolution1
model.add(Conv2D(96, (7, 7), padding="same", input_shape=input_shape))
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.normalization.BatchNormalization(axis=1))

#Convolution2
model.add(Conv2D(256, (5, 5), padding="same", strides=(2, 2)))
model.add(Activation='relu')
model.add(model.add(MaxPooling2D(pool_size=(2, 2))), strides=(2, 2))

#Convolution3
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation='relu')

#Convolution4
model.add(Conv2d(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation='relu')

#Convolution5
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2, 2)))

#FulllyConnectedlLayer6
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.9))
#FullyConnectedLayer7
model.add(Dense(2048, activation='softmax'))
model.add(Dropout(0.9))


sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
