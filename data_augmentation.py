#!/usr/bin/env python
# coding: utf-8

# In[ ]:


conda install keras-gpu


# 
# 
# Data augmentation is a process of creating multiple images of asingle image by changing its paramters like shifting right shifting left rotation zooming fliping it etc.This technique is mainly used when there are less number of data to create multiple images from a single images which are not exactly the same as the original image

# In[3]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('train\cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='data_augmented_image', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# In[1]:


cd


# In[ ]:




