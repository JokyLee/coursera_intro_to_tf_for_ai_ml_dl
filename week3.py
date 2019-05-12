#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

__author__ = 'Li Hao'
__date__ = '2019.05.12'


import tensorflow as tf

# YOUR CODE STARTS HERE
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.998):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()
# YOUR CODE ENDS HERE

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# YOUR CODE STARTS HERE
training_images = training_images.reshape(*training_images.shape, 1)
test_images = test_images.reshape(*test_images.shape, 1)
training_images  = training_images / 255.0
test_images = test_images / 255.0
# YOUR CODE ENDS HERE

model = tf.keras.models.Sequential([
    # YOUR CODE STARTS HERE
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    # YOUR CODE ENDS HERE
])

# YOUR CODE STARTS HERE
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
loss, acc = model.evaluate(test_images, test_labels)
# YOUR CODE ENDS HERE
