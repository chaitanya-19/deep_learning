#!/usr/bin/env python
# coding: utf-8

# # Dog vs Cat

# **Competition Description**

# In this competition, you'll write an algorithm to classify whether images contain either a dog or a cat.  This is easy for humans, dogs, and cats. Your computer will find it a bit more difficult.

# In[1]:


from IPython.display import Image
Image(url= "https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg")


# ## Collecting the Data¶

# training data set and testing data set are given by Kaggle you can download from kaggle directly
# 
# link-https://www.kaggle.com/c/dogs-vs-catsdata

# ## Data Preprocessing

# In[2]:


import numpy as np
import os
from random import shuffle
from PIL import Image ,ImageOps
import pandas as pd


# In[3]:


train_dir='E:\\data\\dog vs cat\\train'


# **We have to extract the target variable if it is cat or dog and label them accordingly**

# In[4]:


def label_images(img):
    word_label=img.split('.')[-3]
    #if word_label == 'dog ': return 1
    if word_label == 'cat': return 0
    else : return 1


# In[5]:


def extract_lables():
    
    train_label=[]
    for img in os.listdir(train_dir):
        label=label_images(img)
        
        train_label.append(np.array(label))

    return train_label
    


# In[6]:


train_lables=extract_lables()
train_lables=np.array(train_lables)
train_lables


# In[7]:


from tensorflow.keras.utils import to_categorical
to_categorical(train_lables)


# **We have extracted the target variable and made a data frame of it**

# In[8]:


df_label=pd.DataFrame(train_lables,columns=['target'])
df_label


# **Now we have to extract the pixel values of data so that we can feed it to the CNN model firstly we resize the image to a 50*50 image then we have to convert it into gray scale after which we convert it into a 1d array for all the images to  feed into our CNN**

# In[9]:


def extract_pixcels():
    train_data=[]
   
    for img in os.listdir(train_dir):
        
        path=os.path.join(train_dir,img)
        img = Image.open(path)
        size=(50,50)
        img= img.resize(size)
        img=ImageOps.grayscale(img)
        img=np.array(img)
        img=img.flatten()
        img=np.array(img)
        train_data.append(img)
      
    
    return train_data


# In[10]:


train_data=extract_pixcels()
train_data=np.array(train_data)
train_data


# We first convert the values of pixels between 0 and 1 by dividing it by 255

# In[11]:


train_data=train_data/255


# In[12]:



x = train_data.reshape(-1, 50, 50, 1)


# **Now we represent the data in form of a data frame so that we can feed it to our CNN**

# In[13]:


import pandas as pd
df=pd.DataFrame(train_data)
df


# In[14]:


final=pd.concat([df, df_label], axis=1)


# In[15]:


final


# NUMBER OF PIXELS =5625
# 
# NUMBER OF SAMPLES=25000

# ## Data Visulization
# 
# 

# In[39]:


#showing an image 
import matplotlib.pyplot as plt
for img in os.listdir(train_dir):
    path=os.path.join(train_dir,img)
    img = Image.open(path)
    plt.imshow(img)
    break


# In[47]:


train_lables


# In[46]:


import seaborn as sns
ax = sns.countplot(x=train_lables)


# ## Creating a Convulutional Neural Network

# In[16]:


#importing all the required libraires

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join


# In[33]:


input_shape=(50,50,1)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten()) # Output convert into one dimension layer and will go to Dense layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[34]:


model.compile(loss='binary_crossentropy', 
              optimizer='Adam',
              metrics=['accuracy'])


# In[35]:


x.shape


# In[36]:


train_lables.shape


# In[37]:


model.fit(x,train_lables,epochs=20)


# We have got an accuraccy of **99** on the training data set

# I didnt split the data initially for testing so we are not able to perform testing

# In[ ]:




