import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras import utils
import h5py
import cv2
import numpy as np
import os


def wtf_getflow(fname, category):
    global count
    cam = cv2.VideoCapture("./"+category + "/" +fname)
    print("./"+category + "/" +fname)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.resize(prevgray, (224, 224))
    videoflow = []
    while ret:
        ret, img = cam.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, cv2.OPTFLOW_FARNEBACK_GAUSSIAN, 0.5, 5, 15, 3, 7, 1.5, 0)
            prevgray = gray
            videoflow.append(flow)
        else:
            break
        array1 = np.concatenate(videoflow, axis=-1)
        #print(array1.shape)
    #print(len(array1), len(data[0]), len(data[0][0]), len(data[0][0][0]))
    #print(array1.T.shape)
    data = []
    for i in range(array1.shape[2]-597):
        data.append(array1[:, :, i:i+600])
    print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]))
    return data
cv2.destroyAllWindows()

cat = ["carcrash", "fight", "gun"]
x_train = []
y_train = []
counter = -1
for category in cat:
    counter += 1
    for file in os.listdir("./"+category)[:1]:
        temp_data = wtf_getflow(file, category)
        for thingie in temp_data:
            x_train.append(thingie)
        #x_train.append(wtf_getflow(file, category))
        print(len(x_train), len(x_train[0]), len(x_train[0][0]), len(x_train[0][0][0]))
        y_train.append(np.array(counter))
#Input Definitions
learning_rate = 0.001
batch_size = 256
num_classes = 3
y_train = keras.utils.to_categorical(y_train, num_classes)
model = Sequential()
input_shape = (224, 224, 598)
#Convolution1
model.add(Conv2D(96, (7, 7), padding="same", input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.normalization.BatchNormalization(axis=1))

#Convolution2
model.add(Conv2D(256, (5, 5), padding="same", strides=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#Convolution3
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))

#Convolution4
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))

#Convolution5
model.add(Conv2D(512, (3, 3), padding="same", strides=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#FulllyConnectedlLayer6
model.add(Dropout(0.9))
model.add(Dense(4096, activation='relu'))


#FullyConnectedLayer7
model.add(Dropout(0.9))
model.add(Dense(2048, activation='softmax'))

#FullyConnectedLayer8
model.add(Dropout(0.9))
model.add(Dense(1048, activation='softmax'))

#FullyConnectedLayer9
model.add(Dropout(0.9))
model.add(Dense(100, activation='softmax'))

#FullyConnectedLayer10
model.add(Dropout(0.9))
model.add(Dense(3, activation='softmax'))



ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss="sparse_categorical_crossentropy", optimizer='Adadelta', metrics=['accuracy'])

x_train = np.array(x_train)
y_train = y_train.reshape((-1, 1))
for i in range(1, 10):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=10, shuffle=True)
    model.save("F:/smackdown/"+"iteration_"+str(i)+".h5")
