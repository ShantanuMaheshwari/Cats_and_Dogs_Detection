# !usr/bin/python3
"""Cats and Dogs classifier
Author: Shantanu Maheshwari
Date: 2019-08-16
"""

#%%

import numpy as np
import keras
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import gc

#%%

os.chdir(r'/PythonProjects/Cats_and_Dogs_Detection/')
print(os.listdir('input'))

#%%

train_dir = 'input/train'
test_dir = 'input/test'
# dog images
train_dogs = ['input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]
# cat images
train_cats = ['input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]

# get test images
test_imgs = ['input/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_imgs)

#%%

print(len(train_dogs))
print(len(train_cats))
print(len(test_imgs))

# del(train_dogs)
# del(train_cats)
# gc.collect()

#%%

# Counting number of files in the extracted directory
path, dirs, files = next(os.walk("input/train"))
file_count = len(files)
print(file_count)

#%%

for ima in train_imgs[0:3]:
    img = mpimg.imread(ima)
    implot = plt.imshow(img)
    plt.show()

#%%

nrows, ncols = 150, 150
# channel 1 is for greyscale
channel = 3

#%%

# print(train_imgs[0])
(cv2.imread(train_imgs[0])).shape

#%%

def read_and_procecss_images(list_of_images):
    X = []   #images
    y = []   #labels

    for img in list_of_images:
        X.append(cv2.resize(cv2.imread(img, cv2.IMREAD_COLOR), (nrows,ncols), interpolation = cv2.INTER_CUBIC))  #read image
        if 'dog' in img:
            y.append(1)
        elif 'cat' in img:
            y.append(0)
    return X, y


X, y = read_and_procecss_images(train_imgs)

#%%

plt.figure(figsize = (20,10))
cols = 5
for i in range(cols):
    plt.subplot(5/cols +1, cols, i+1)
    plt.imshow(X[i])

#%%

