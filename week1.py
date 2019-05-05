#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

__author__ = 'Li Hao'
__date__ = '2019.05.05'


import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1, 0, 1, 2, 3, 4, 10], dtype=np.float)
ys = np.array([0, 0.5, 1, 1.5, 2, 2.5, 5.5], dtype=np.float)
model.fit(xs, ys, epochs=1000)
print(model.predict([7.0]))
