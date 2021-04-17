import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout
from keras.callbacks import  ModelCheckpoint
import random

def augument(img):
  centre = (25, 25)
  M = cv2.getRotationMatrix2D(centre, np.random.randint(-10,11), 1)
  img = cv2.warpAffine(img, M, (50,50))
  return(img)

path = '/content/drive/MyDrive/emotion_classifier/dataset_new.npy'
dataset = np.load(path, allow_pickle = True)
random.shuffle(dataset)

train_inputs = []
train_targets = []

# To append the images and targets corresspondingly
for img, target in dataset:
  train_inputs.append(augument(img))
  train_targets.append(target)

train_inputs = np.array(train_inputs)
train_targets = np.array(train_targets)

# Normalisation
normalised_train_inputs = train_inputs/255

model = keras.Sequential()

# CADM - 32
model.add(Conv2D(32, (3,3), padding = 'same', input_shape = normalised_train_inputs.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2), strides = (2,2)))

# CADM - 64
model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2), strides = (2,2)))

# CADM - 128
model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2), strides = (2,2)))

# CADM - 256
model.add(Conv2D(256, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D((2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(4))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(5)) # 1 output
model.add(Activation('softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy'])

# Training the model
file_path = '/content/drive/MyDrive/emotion_classifier/facial_emotion_classifier_model_new.hdf5'
check_point = ModelCheckpoint(file_path, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [check_point]
model.fit(normalised_train_inputs, train_targets, validation_split = 0.05, batch_size=100, epochs = 50, callbacks = callbacks_list, verbose = 1)

model.summary()