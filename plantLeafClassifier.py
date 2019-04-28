#!/usr/bin/env python
# coding: utf-8

# In[152]:


import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# In[153]:


#The dataset used in the project is plantvillage dataset that contains fifteen classes of diseases corresponding to potato, bell pepper and tomato plants


# In[154]:


DATADIR = "C:\Datasets"
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
width=256
height=256
depth=3


# In[155]:


#DefineClassLabels
class_labels = ['bell_pepper_bacterial_spot', 'bell_pepper_healthy','potato_early_blight','potato_healthy','potato_late_blight','tomato_target_spot','tomato_mosaic_virus'
                ,'tomato_yellow_leaf','tomato_bacterial_spot','tomato_early_blight','tomato_healthy','tomato_late_blight','tomato_leaf_mold'
               ,'tomato_septoria_leaf_spot','tomato_spider_mites']
print(default_image_size)


# In[156]:


for labels in class_labels:
    path = os.path.join(DATADIR,labels)
    class_num = class_labels.index(labels)
    for img in os.listdir(path):
        img = cv2.imread(os.path.join(path,img))
        plt.imshow(img) 
        plt.show()
        break
    break
        


# In[157]:


#loading dataset into the kernel
img_list = []
for labels in class_labels:
    path = os.path.join(DATADIR,labels)
    class_num = class_labels.index(labels)
    for img in os.listdir(path):
        image = cv2.resize(cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) , (15 , 15))
        plt.imshow(image) 
        plt.show()
        img_list.append([image , class_num ])
        break
    break
        
print(image)
       


# In[158]:


import random


# In[159]:


random.shuffle(img_list)


# In[160]:


label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(class_labels)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)


# In[161]:


print(label_binarizer.classes_)


# In[179]:


X = []
y = []
X_test = []
Y_test = []

for features,label in img_list:
    X.append(features)
    y.append(label)
for features,label in img_list[:30]:
    X_test.append(features)
    Y_test.append(label)
print(X[0].reshape(-1, 15, 15, 1))
print()
print('Test Arrays')
print()
print(X_test[0].reshape(-1,15,15,1))

X = np.array(X).reshape(-1, 15, 15, 1)

X_t = np.array(X_test).reshape(-1,15,15,1)


# In[ ]:





# In[163]:


img_array.shape


# In[164]:


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
image_size = 0
width=256
height=256
depth=3


# In[165]:


#Split the Dataset into Train and Test Samples
#train - 80%
#test -20%
print('Splitting the model')
X_train, X_test, Y_train, Y_test = train_test_split(np_img_list, image_labels, test_size=0.2, random_state = 42) 
X_t, X_test, Y_t, Y_test = train_test_split(np_img_list, image_labels, test_size=0.2, random_state = 42) 
print('Dataset Split Successfully')


# In[166]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
X = X/255.0


# In[167]:


model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('relu'))


# In[168]:


model.summary()


# In[169]:


op = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=op,metrics=["accuracy"])


# In[170]:


#training our neural network
history = model.fit(X, y, batch_size=32, epochs=10)


# In[174]:


#testAccuracy

history = model.fit(X_t, Y_test, batch_size=5, epochs=100)


# In[ ]:





# In[ ]:





# In[ ]:




