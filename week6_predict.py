#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

__author__ = 'Li Hao'
__date__ = '2019.05.30'


# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import cv2
import sys
import math
import zipfile
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile, rmtree
from tensorflow.keras.preprocessing import image


_cur_dir = os.path.abspath(os.path.join(__file__, "../"))
model = tf.keras.models.load_model(os.path.join(_cur_dir, "data/models/cat_vs_dog/week6_cat_v_dog_model"))
test_dir = os.path.join(_cur_dir, "data/cat_v_dog/src_kaggle/test1")
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
for f in os.listdir(test_dir):
    path = os.path.join(test_dir, f)
    img = cv2.imread(os.path.join(test_dir, f))
    img = cv2.resize(img, (150, 150))
    x = np.expand_dims(img, axis=0)
    classes = model.predict(x, batch_size=10)
    print(classes[0])

    if classes[0] > 0.5:
        print(f + " is a dog")
        animal_class = "dog"
    else:
        print(f + " is a cat")
        animal_class = "cat"

    cv2.putText(img, animal_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imshow("img", img)
    char = chr(cv2.waitKey(0) & 255)
    if char == 'q':
        break
