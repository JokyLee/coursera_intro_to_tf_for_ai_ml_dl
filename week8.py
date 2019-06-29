#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

__author__ = 'Li Hao'
__date__ = '2019.06.28'


import os
import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder


_cur_dir = os.path.abspath(os.path.join(__file__, "../"))


def get_data(filename):
    # You will need to write code that will read the file passed
    # into this function. The first line contains the column headers
    # so you should ignore it
    # Each successive line contians 785 comma separated values between 0 and 255
    # The first value is the label
    # The rest are the pixel values for that picture
    # The function will return 2 np.array types. One with all the labels
    # One with all the images
    #
    # Tips:
    # If you read a full line (as 'row') then row[0] has the label
    # and row[1:785] has the 784 pixel values
    # Take a look at np.array_split to turn the 784 pixels into 28x28
    # You are reading in strings, but need the values to be floats
    # Check out np.array().astype for a conversion
    labels = []
    images = []
    with open(filename) as file:
        _ = file.readline()
        for col in file:
            data = list(map(int, col.split(",")))
            labels.append(data[0])
            images.append(np.array_split(data[1:], 28))
    # enc = OneHotEncoder()
    # labels = enc.fit_transform(labels).toarray()
    labels = np.array(labels)
    # Your code starts here
    # Your code ends here
    return np.array(images, dtype=np.float), np.array(labels)


training_path = os.path.join(_cur_dir, 'data/sign-language-mnist/sign_mnist_train.csv')
testing_path = os.path.join(_cur_dir, 'data/sign-language-mnist/sign_mnist_test.csv')
training_images, training_labels = get_data(training_path)
testing_images, testing_labels = get_data(testing_path)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# Their output should be:
# (27455, 28, 28)
# (27455,)
# (7172, 28, 28)
# (7172,)

# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims

training_images = training_images.reshape(*training_images.shape, 1) # Your Code Here
testing_images = testing_images.reshape(*testing_images.shape, 1) # Your Code Here

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    # Your Code Here
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    # Your Code Here
    rescale = 1./255,
)

# Keep These
print(training_images.shape)
print(testing_images.shape)

# Their output should be:
# (27455, 28, 28, 1)
# (7172, 28, 28, 1)

# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D
model = tf.keras.models.Sequential([
    # Your Code Here
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(24, activation=tf.nn.softmax)
])

print(model.summary())
# Compile Model.
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

enc = OneHotEncoder()
training_labels = enc.fit_transform(training_labels.reshape(-1, 1)).toarray()
all_index = set(range(training_images.shape[0]))
validation_idx = random.sample(all_index, int(training_images.shape[0] * 0.1))
training_idx = list(all_index.difference(set(validation_idx)))
validation_images = training_images[validation_idx]
validation_labels = training_labels[validation_idx]
training_images = training_images[training_idx]
training_labels = training_labels[training_idx]

# Train the Model
history=model.fit_generator(
    train_datagen.flow(training_images, training_labels),
    epochs=25,
    validation_data=validation_datagen.flow(validation_images, validation_labels),
    verbose=1
)

testing_labels = enc.fit_transform(testing_labels.reshape(-1, 1)).toarray()
model.evaluate(testing_images, testing_labels)

import pickle
SAVED_PATH = os.path.join(_cur_dir, "data/models/sign_language")
os.makedirs(SAVED_PATH, exist_ok=True)
model.save(os.path.join(SAVED_PATH, "sign_language.h5"))
with open(os.path.join(SAVED_PATH, "sign_language_history"), "wb") as f:
    pickle.dump(history.history, f)

# The output from model.evaluate should be close to:
# [6.92426086682151, 0.56609035]

# Plot the chart for accuracy and loss on both training and validation

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()