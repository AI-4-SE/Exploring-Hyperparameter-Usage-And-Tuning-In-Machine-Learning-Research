#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import os
from PIL import Image


import IPython.display 
from keras.preprocessing.image import array_to_img 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from keras.callbacks import TensorBoard 
from time import strftime 


# # Constants

# In[ ]:


IMAGES_PATH = '../input/plant-pathology-2020-fgvc7/images' 
TRAIN_PATH =  '../input/plant-pathology-2020-fgvc7/train.csv'
TEST_PATH =  '../input/plant-pathology-2020-fgvc7/test.csv' 


IMG_WIDTH = IMG_HEIGHT = 300 
NR_CHANNELS = 3
TOTAL_INPUTS = NR_CHANNELS * IMG_HEIGHT * IMG_WIDTH

LOG_DIR = '/kaggle/working/tensorboard_plant_logs/'

COLUMNS = ["healthy", "multiple_diseases", "rust", "scab"]
LABEL_ENCODE = np.array([0,1,2,3])


# # Get the Data

# In[ ]:


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH) 


# In[ ]:


train.shape


# In[ ]:


train.head()


# ### In train set; 
# * image_id: the foreign key for the parquet files
# * combinations: one of the target labels
# * healthy: one of the target labels
# * rust: one of the target labels
# * scab: one of the target labels

# In[ ]:


train.tail()


# In[ ]:


train.info() 


# In[ ]:


train['healthy'].value_counts()


# In[ ]:


train['multiple_diseases'].value_counts()


# In[ ]:


train['rust'].value_counts()


# In[ ]:


train['scab'].value_counts()


# ## Get the imgs

# * As you can see, we have large images. 
# * We can't load all of them in main memory. 
# * There is several ways to handle this problem but we just resize it. 
# * For now its size is going to be fixed, at the end we can arrange it again. 

# In[ ]:


ex_image = os.path.join(IMAGES_PATH, train.image_id[0] + '.jpg')
ex_image = np.asanyarray(Image.open(ex_image)) 
ex_image.shape


# In[ ]:


def load_images(dataset, folder): 
    arr = [] 
    for imageName in dataset:
        path = os.path.join(folder, imageName + '.jpg') 
        img = Image.open(path) 
        img = img.resize((300, 300)) 
        arr.append(np.asanyarray(img)) 
        img.close() 
    
    return arr 


# In[ ]:


train_images_all = np.array(load_images(train.image_id, IMAGES_PATH)) 
train_images_all.shape


# In[ ]:


test_images_all = np.array(load_images(test.image_id, IMAGES_PATH))
test_images_all.shape


# In[ ]:


# get target from train 
train_all_y = train.iloc[:, 1:5]


# In[ ]:


# Split data for validation set and flatten data 
x_train = train_images_all[:1638] 
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = train_all_y[:1638] 
y_train = np.dot(y_train, LABEL_ENCODE.T) # to label encoding

x_val = train_images_all[1638:] # %10 for validation set 
x_val = x_val.reshape(x_val.shape[0], -1)
y_val = train_all_y[1638:]
y_val = np.dot(y_val, LABEL_ENCODE.T) # to label encoding

print("x_val shape: " , x_val.shape) 
print("x_train shape: ",x_train.shape)


# In[ ]:


plt.figure(figsize=(15,7)) 
for i in range(1,10): 
    plt.subplot(1, 10, i) 
    plt.yticks([])
    plt.xticks([]) 
    plt.imshow(train_images_all[i]) 


# # Pre-process Data 

# In[ ]:


type(train_images_all[0][0][0][0])


# In[ ]:


train_images_all, test_images_all = train_images_all / 255.0, test_images_all / 255.0


# In[ ]:


print(type(train_images_all[0][0][0][0])) 
print(train_images_all[0][0][0][0])


# In[ ]:


# Flat Images to one single vector 
train_images_all = train_images_all.reshape(len(train_images_all), TOTAL_INPUTS)
test_images_all = test_images_all.reshape(len(test_images_all), TOTAL_INPUTS)


# In[ ]:


train_images_all.shape


# # Define the Neural Network Using Keras 

# In[ ]:


model_1 = Sequential([
    Dense(units=32, input_dim=TOTAL_INPUTS, activation='relu'),  
    Dense(units=16, activation='relu'), 
    Dense(units=4, activation='softmax') 
])


# In[ ]:


type(model_1)


# In[ ]:


model_1.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
model_1.summary() 


# # Tensorboard

# In[ ]:


def get_tensorboard(model_name): 
    folder_name = f'{model_name} at {strftime("%H %M")}'
    dir_paths = os.path.join(LOG_DIR, folder_name) 
    
    try: 
        os.makedirs(dir_paths) 
    except OSError as err:
        print(err.strerror)
    else: 
        print('Succesfully created.') 
    
    return TensorBoard(log_dir=dir_paths), dir_paths 


# # Fit the Model 

# In[ ]:


samples_per_batch = 200 
nr_epoch = 100


# In[ ]:


get_ipython().run_cell_magic('time', '', "tensorboard, logdir = get_tensorboard('model_1') \nhistory_model_1 = model_1.fit(x_train, y_train, batch_size=samples_per_batch, validation_data=(x_val, y_val), epochs= nr_epoch, \n           callbacks=[tensorboard], verbose=0)\n")


# In[ ]:


def display_model_hist(train_loss, train_accuracy, val_loss, val_accuracy): 
    
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    axs[0,0].set_xlabel('Nr of Epochs', fontsize=18)
    axs[0,0].set_ylabel('Train Loss', fontsize=18)
    axs[0,0].plot(train_loss,color='green')

    axs[0,1].set_xlabel('Nr of Epochs', fontsize=18)
    axs[0,1].set_ylabel('Train Accuracy', fontsize=18)
    axs[0,1].plot(train_accuracy, color='green') 
    
    axs[1,0].set_xlabel('Nr of Epochs', fontsize=18)
    axs[1,0].set_ylabel('Val Loss', fontsize=18)
    axs[1,0].plot(val_loss,color='magenta')

    axs[1,1].set_xlabel('Nr of Epochs', fontsize=18)
    axs[1,1].set_ylabel('Val Accuracy', fontsize=18)
    axs[1,1].plot(val_accuracy, color='magenta') 
    

    plt.show()


# In[ ]:


loss = history_model_1.history['loss'] 
accs = history_model_1.history['accuracy'] 

val_loss = history_model_1.history['val_loss']
val_accuracy = history_model_1.history['val_accuracy']

display_model_hist(loss, accs, val_loss, val_accuracy) 


# * Try early stopping, because we have obtained high accuracy and low loss previous epoch.  

# In[ ]:


nr_epoch = 200


# In[ ]:


get_ipython().run_cell_magic('time', '', "tensorboard, logdir = get_tensorboard('model_2') \nhistory_model_2 = model_1.fit(x_train, y_train, batch_size=samples_per_batch, validation_data=(x_val, y_val), epochs= nr_epoch, \n           callbacks=[tensorboard], verbose=0)\n")


# In[ ]:


loss = history_model_2.history['loss'] 
accs = history_model_2.history['accuracy'] 

val_loss = history_model_2.history['val_loss']
val_accuracy = history_model_2.history['val_accuracy']

display_model_hist(loss, accs, val_loss, val_accuracy) 


# In[ ]:


model_3 = Sequential([
    Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)),
    Dense(units=64, activation='relu'),  
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'), 
    Dense(units=4, activation='softmax') 
])

model_3.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
model_3.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "tensorboard, logdir = get_tensorboard('model_3') \nhistory_model_3 = model_3.fit(x_train, y_train, batch_size=samples_per_batch, validation_data=(x_val, y_val), epochs= nr_epoch, \n           callbacks=[tensorboard], verbose=0)\n")


# In[ ]:


loss = history_model_3.history['loss'] 
accs = history_model_3.history['accuracy'] 

val_loss = history_model_3.history['val_loss']
val_accuracy = history_model_3.history['val_accuracy']

display_model_hist(loss, accs, val_loss, val_accuracy) 


# # Prediction and Evaluate Model

# In[ ]:


test_ph = np.expand_dims(test_images_all[0], axis=0)
test_ph.shape


# In[ ]:


model_3.predict(test_ph)


# In[ ]:


test_images_all.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'result = model_3.predict(test_images_all)\n')


# In[ ]:


submission = pd.DataFrame(columns=train.columns[1:], data=result)
submission = pd.concat([test, submission], axis=1)


# In[ ]:


submission.head() 

