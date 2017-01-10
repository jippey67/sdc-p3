
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import json

dataDir = 'data/'
autodata = dataDir + 'driving_log.csv'
steer_correct = 0.1
steer_threshold = 0.1
steer_keep_prob = 0.3
paths = ['IMG/center_','IMG/left_','IMG/right_']

def train_gen(set,batchsize=64):
    timestamp, steer = shuffle_set(set)
    i=0
    while 1:
        X_batch=[]
        y_batch=[]
        j = batchsize
        while j>0:
            if i == len(timestamp):
                i=0
                timestamp, steer = shuffle_set(set)
            steer_ang = steer[i]

            # randomly choose camview : center, left or right camera
            cam_view = np.random.randint(3)
            if cam_view == 1:
                steer_ang += steer_correct
            if cam_view == 2:
                steer_ang -= steer_correct

            # randomly choose whether an image with an associated steering angle lower than steer_threshold
            # will be included in the batch. To address bias for driving straight
            if np.absolute(steer_ang)<steer_threshold:
                if np.random.uniform() >= steer_keep_prob:
                    continue
            pad = dataDir+paths[cam_view]+timestamp[i]+'.jpg'
            img = adapt_image(pad)

            # flip the image around y-axis 50% probability
            if np.random.randint(2) == 1:
                img = cv2.flip(img, 1)
                steer_ang = -steer_ang

            # normalize the image in range [-0.5, 0.5]
            img = norm_image(img)
            #plt.imshow(img, cmap='gray')
            #plt.show()
            #print (img.shape)

            X_batch.append(img)
            y_batch.append(steer_ang)
            i += 1
            j -= 1
        yield np.reshape(np.array(X_batch),(batchsize,64,64,3)), np.array(y_batch)

def val_gen(set,batchsize=64):
    timestamp, steer = shuffle_set(set)
    i = 0
    while 1:
        X_batch = []
        y_batch = []
        j = batchsize
        while j > 0:
            if i == len(timestamp):
                i=0
                timestamp, steer = shuffle_set(set)
            steer_ang = steer[i]
            pad = dataDir + paths[0] + timestamp[i] + '.jpg'
            img = adapt_image(pad)
            img = norm_image(img)
            X_batch.append(img)
            y_batch.append(steer_ang)
            i += 1
            j -= 1
        yield np.reshape(np.array(X_batch),(batchsize,64,64,3)), np.array(y_batch)

def norm_image(img):
    img = img.astype(float)
    img = (img / 255) - 0.5
    return img

def shuffle_set(set):
    np.random.shuffle(set)
    timestamp = set[:, 0]
    steer = set[:, 1].astype(float)
    return timestamp, steer

def adapt_image(pad):
    # remove irrelevant pixels from image (top 60, bottom 20) and convert to HSV color space
    img = plt.imread(pad)[60:140, :]
    img = cv2.cvtColor(plt.imread(pad), cv2.COLOR_RGB2HSV)[60:140, :]
    # resize to square for easier use in convolutions
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return img

def define_model():
    model = Sequential()
    # input: 64x64 images with 1 channel -> (64, 64) tensors.
    # this applies 24 convolution filters of size 3x3 each.
    model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), input_shape=(64,64,3), activation='relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(8, 3, 3, border_mode='valid', subsample=(2,2)))
    model.add(Convolution2D(4, 3, 3, activation='relu'))
    model.add(Convolution2D(2, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model

def save_model(model):
    from keras.models import model_from_json
    model_json = model.to_json()
    with open("model.json", "w") as f:
        json.dump(model_json,f)
    model.save_weights("model.h5")

steer_angle = []
with open(autodata) as csvfile:
    lines = csvfile.read().split("\n")  # "\r\n" if needed
    for line in lines:
        if (line[0:3] == "IMG") or (line[0:3] == "/Us"):
            cols = line.split(",")
            steer_angle.append( [ (cols[0][len(cols[0]) - 27:-4:]), float(cols[3])] )
steer_angle = np.array(steer_angle)
np.random.shuffle(steer_angle)
num_samples = len(steer_angle)
train_set = steer_angle[:int(0.9*num_samples)]
val_set = steer_angle[int(0.9*num_samples):]

tr_gen = train_gen(train_set, batchsize=64)
#X_train ,y_train = next(tr_gen)

vl_gen = val_gen(val_set, batchsize=256)
#X_val, y_val = next(vl_gen)

model = define_model()
model.fit_generator(generator=tr_gen, samples_per_epoch=len(steer_angle), nb_epoch=10, validation_data=vl_gen, nb_val_samples=64)
save_model(model)

