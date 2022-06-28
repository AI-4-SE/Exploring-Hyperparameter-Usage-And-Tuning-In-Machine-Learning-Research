#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from PIL import Image

# Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


train_folder= '../input/chest-xray-pneumonia/chest_xray/train/'
val_folder = '../input/chest-xray-pneumonia/chest_xray/val/'
test_folder = '../input/chest-xray-pneumonia/chest_xray/test/'

# train 
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'

cnn = Sequential()

#Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

#Pooling
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd Convolution
cnn.add(Conv2D(32, (3, 3), activation="relu"))

# 2nd Pooling layer
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the layer
cnn.add(Flatten())

# Fully Connected Layers
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'sigmoid', units = 1))

# Compile the Neural network
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(train_folder,
                                           target_size = (64, 64),
                                           batch_size = 32,
                                           class_mode = 'binary')

validation_set = test_datagen.flow_from_directory(val_folder,
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

test_set = test_datagen.flow_from_directory(test_folder,
                                       target_size = (64, 64),
                                       batch_size = 32,
                                       class_mode = 'binary')


cnn_model = cnn.fit(training_set,
                    steps_per_epoch = 163,
                    epochs = 5,
                    validation_data = validation_set,
                    validation_steps = 624)

test_accu = cnn.evaluate(test_set,steps=624)

print('The testing accuracy is :',test_accu[1]*100, '%')







