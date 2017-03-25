"""
spatial cnn that classifies evry static frame in the inputted video. built on top of the VGG16 network trained on ImageNet
"""

from keras.models import Sequential
from keras.layer import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import os

#for all frames in one video
def get_frame(fname, category):
 arr1=[]
 cap=cv2.VideoCapture("./"+category + "/"+fname)
 success,frame=cam.read()
 frame1=cv2.resize(frame, (140,140))
 success=True
 while success:
  success,frame=cap.read()
  frame1=cv2.resize(frame, (140,140))
  arr1.append(frame1)
 return arr1

cat=["carcrash", "fight", "gun"]
X_train=[]
Y_train=[]

for cat in category:
 for file in os.listdir("./"+category)[:1]:
  temp=[]
  temp=get_frame(file,cat)
  X_train.append(temp)


#building the base vgg model such that features are extracted using the vggmodel. till the end of the convulution layers
base_model=VGG16(weights='imagenet', include_top=False)
x=preprocess_input(x)

features=base_model.predict(x)
np.save(open('features.npy', 'w'),features)

#building the bottleneck
x_train=np.load(open('features.npy'))

top_model = Sequential()
top_model.add(Flatten())
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.9))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.8))
top_model.add(Dense(20,activation='softmax'))



top_model.compile(optimizer='Adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

top_model.fit( x_train, y_train, batch_size=256, epochs=50,validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
top_model.save('spatial_cnn.h5)
