#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

__author__ = 'Li Hao'
__date__ = '2019.05.05'


import tensorflow as tf

# YOUR CODE SHOULD START HERE
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # if(logs.get('loss')<0.4):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()
# YOUR CODE SHOULD END HERE

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# YOUR CODE SHOULD START HERE
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()
# print(y_train[0])
# print(x_train[0])
x_train  = x_train / 255.0
x_test = x_test / 255.0
# YOUR CODE SHOULD END HERE

model = tf.keras.models.Sequential([
    # YOUR CODE SHOULD START HERE
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    # YOUR CODE SHOULD END HERE
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#
# # YOUR CODE SHOULD START HERE
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
loss, acc = model.evaluate(x_test, y_test)
# # YOUR CODE SHOULD END HERE