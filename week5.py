#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

__author__ = 'Li Hao'
__date__ = '2019.05.30'


# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import sys
import math
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile, rmtree

# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
#
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list
def copyFiles(srcDir, dstDir, files):
    os.makedirs(dstDir, exist_ok=True)
    for f in files:
        src_path = os.path.join(srcDir, f)
        dst_path = os.path.join(dstDir, f)
        copyfile(src_path, dst_path)


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # YOUR CODE STARTS HERE
    assert 0. <= SPLIT_SIZE <= 1.
    source_files = os.listdir(SOURCE)
    filtered_source_files = []
    for f in source_files:
        if os.path.getsize(os.path.join(SOURCE, f)):
            filtered_source_files.append(f)
        else:
            print("{} is zero length, so ignoring".format(f))

    training_files = random.sample(filtered_source_files, math.floor(len(filtered_source_files) * SPLIT_SIZE))
    copyFiles(SOURCE, TRAINING, training_files)

    testing_files = set(filtered_source_files).difference(set(training_files))
    copyFiles(SOURCE, TESTING, testing_files)
    # YOUR CODE ENDS HERE


_cur_dir = os.path.abspath(os.path.join(__file__, "../"))
CAT_SOURCE_DIR = os.path.join(_cur_dir, "data/cat_v_dog/kagglecatsanddogs_3367a/PetImages/Cat/")
TRAINING_CATS_DIR = os.path.join(_cur_dir, "data/cat_v_dog/training/cats/")
TESTING_CATS_DIR = os.path.join(_cur_dir, "data/cat_v_dog/testing/cats/")
DOG_SOURCE_DIR = os.path.join(_cur_dir, "data/cat_v_dog/kagglecatsanddogs_3367a/PetImages/Dog/")
TRAINING_DOGS_DIR = os.path.join(_cur_dir, "data/cat_v_dog/training/dogs/")
TESTING_DOGS_DIR = os.path.join(_cur_dir, "data/cat_v_dog/testing/dogs/")

OVERWRITE = False
split_size = .9
if OVERWRITE:
    rmtree(TRAINING_CATS_DIR, ignore_errors=True)
    rmtree(TRAINING_DOGS_DIR, ignore_errors=True)
    rmtree(TESTING_CATS_DIR, ignore_errors=True)
    rmtree(TESTING_DOGS_DIR, ignore_errors=True)
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring

print(len(os.listdir(TRAINING_CATS_DIR)))
print(len(os.listdir(TRAINING_DOGS_DIR)))
print(len(os.listdir(TESTING_CATS_DIR)))
print(len(os.listdir(TESTING_DOGS_DIR)))

# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
# USE AT LEAST 3 CONVOLUTION LAYERS
model = tf.keras.models.Sequential([
# YOUR CODE HERE
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics = ['acc']
)

TRAINING_DIR = os.path.join(_cur_dir, "data/cat_v_dog/training") #YOUR CODE HERE
train_datagen = ImageDataGenerator(rescale=1.0/255.) #YOUR CODE HERE
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, batch_size=20, class_mode='binary', target_size=(150, 150)
)   #YOUR CODE HERE

VALIDATION_DIR = os.path.join(_cur_dir, "data/cat_v_dog/testing") #YOUR CODE HERE
validation_datagen = ImageDataGenerator(rescale=1.0/255.) #YOUR CODE HERE
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, batch_size=20, class_mode='binary', target_size=(150, 150)
)
#YOUR CODE HERE

history = model.fit_generator(
    train_generator, epochs=15, verbose=1, validation_data=validation_generator
)

# PLOT LOSS AND ACCURACY
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)

os.makedirs("data/models/cat_vs_dog")
model.save(os.path.join(_cur_dir, "data/models/cat_vs_dog/cat_v_dog"))