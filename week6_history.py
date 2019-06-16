#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

__author__ = 'Li Hao'
__date__ = '2019.05.30'

# PLOT LOSS AND ACCURACY
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
import pickle
import os
_cur_dir = os.path.abspath(os.path.join(__file__, "../"))
with open(os.path.join(_cur_dir, "data/models/cat_vs_dog/week6_cat_v_dog_history"), "rb") as f:
    history = pickle.load(f)
acc=history['acc']
val_acc=history['val_acc']
loss=history['loss']
val_loss=history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.legend(loc=0)
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc=0)
plt.title('Training and validation loss')
plt.show()