"""
spatial cnn that classifies evry static frame in the inputted video. built on top of the VGG16 network trained on ImageNet
"""

from keras.models import Sequential
from keras.layer import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from imagenet_utils import preprocess_input
import numpy as np
import os

#data
"""
 extract images from every frame of a video. store seperately. access.
 convert img_to_array.
 store x and y_train
"""

#building the base vgg model such that features are extracted using the vggmodel. till the end of the convulution layers
base_model=VGG16(weights='imagenet', include_top=False)
x=preprocess_input(x)

features=base_model.predict(x)
np.save(open('features.npy', 'w'),features)

#building the bottleneck
x_train=np.load(open('features.npy'))

top_model = Sequential()
top_model.add(Flatten(input_shape=x_train.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

top_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

top_model.fit( x_train, y_train, batch_size=16, epochs=50,validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

