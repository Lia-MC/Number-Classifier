import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# set x_train to images
# set y_train to labels of images
# set x_test to testing images
# set y_test to labels for testing images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize! -> set it to a value between 0 to 1, 0 = black, 1 = white
x_train - x_train / 255.0
y_train - y_train / 255.0

# define model using keras
model = keras.Sequential([
    # PROCESSES IMAGE
    # 28x28 image is now a 784 elem vector -> each elem is a px
    keras.layers.Flatten(input_shape=(28, 28)) 

    # "LEARNS" THE PATTERNS
    # 128 neurons in a layer in which each neuron connects to every neuron in prev. layer
    # each px is multiplied by a weight and added some bias, and this is passed into relu activation function
    # relu function sets all neg values to 0
    keras.layers.Dense(128, activation='relu') 

    # GENERATES PROBABILITIES
    # 10 neurons in a layer in which each neuron connects to every neuron in prev. layer
    # each neuron represents a number from 0 to 9
    # values are passed into softmax activation function to generate probabilities
    keras.layers.Dense(10, activation='softmax')

    # the value with the highest probability is the number predicted by the model
])
