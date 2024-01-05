# -*- coding: utf-8 -*-
"""
Created on Sun April 18 2021

@author: Daniel

Minimum requirements:
    Python version     : 3.7.7
    TensorFlow version : 2.1.0
    Keras version      : 2.3.1

USAGE:
--model best_model_lbs.h5 --trainvh SAMPLES_ALL_VH.CSV --trainvv SAMPLES_ALL_VV.CSV --layers 3 --epoch 50 --batch 32

26 Dec 2021 (v11):
    - update with normalization (data range is from 0 dB to -30 dB)
    - LSTM hyperparameters can now be specified in the program parameters

27 Dec 2021 (v12):
    - numpy compressed archive format (NPZ) is used instead of CSV
      much smaller size and faster process.

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
import argparse

def readucr(band1, band2):
    data1 = np.load(band1)['arr_0']
    data2 = np.load(band2)['arr_0']
    
    y = data1[:, 0]

    # add one more axis to allow additional dimension for multivar
    x1 = data1[:, 1:, np.newaxis]
    x2 = data2[:, 1:, np.newaxis]

    x = np.append(x1, x2, axis=2)

    return x, y.astype(int)

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

    
# show information
print("Python version     : " + sys.version)
print("TensorFlow version : " + tf.__version__)
print("Keras version      : " + tf.keras.__version__)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-th", "--trainvh", required=True,
	help="path to training samples VH")
ap.add_argument("-tv", "--trainvv", required=True,
	help="path to training samples VV")
ap.add_argument("-ld", "--layers", required=True,
	help="how many layers of NN used")
ap.add_argument("-ep", "--epoch", required=True,
	help="how many epoch will be run")
ap.add_argument("-bs", "--batchsize", required=True,
	help="define batch size")

args = vars(ap.parse_args())

modelpath = args["model"]
trainvhpath = args["trainvh"]
trainvvpath = args["trainvv"]
layers = int(args["layers"])
epochs = int(args["epoch"])
batch_size = int(args["batchsize"])

# load training data
print("Loading training dataset...")
x_all, y_all = readucr(trainvhpath, trainvvpath)

# split dataset for REAL testing.
print("Splitting training & test...")
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.1,
                                            shuffle=True, random_state=42)


# Visualize the data
# Here we visualize one timeseries example for each class in the dataset.

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    avg_cx_train = np.average(c_x_train, axis = 0)
    plt.plot(avg_cx_train, label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()

# [1:]

"""
## Standardize the data

Our timeseries are already in a single length (176). However, their values are
usually in various ranges. This is not ideal for a neural network;
in general we should seek to make the input values normalized.
For this specific dataset, the data is already z-normalized: each timeseries sample
has a mean equal to zero and a standard deviation equal to one. This type of
normalization is very common for timeseries classification problems, see
[Bagnall et al. (2016)](https://link.springer.com/article/10.1007/s10618-016-0483-9).

Note that the timeseries data used here are univariate, meaning we only have one channel
per timeseries example.
We will therefore transform the timeseries into a multivariate one with one channel
using a simple reshaping via numpy.
This will allow us to construct a model that is easily applicable to multivariate time
series.
"""

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 2))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 2))

"""
Finally, in order to use `sparse_categorical_crossentropy`, we will have to count
the number of classes beforehand.
"""

num_classes = len(np.unique(y_train))

"""
Now we shuffle the training set because we will be using the `validation_split` option
later when training.
"""

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


"""
Standardize the labels to positive integers.
The expected labels will then be 0 and 1.
"""

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def make_model1(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def make_model2(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def make_model3(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))    
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if layers == 1:
    model = make_model1(input_shape=x_train.shape[1:])
elif layers == 2:
    model = make_model2(input_shape=x_train.shape[1:])
elif layers == 3:
    model = make_model3(input_shape=x_train.shape[1:])
else:
    print("Invalid layers specified, defaulting to 1 layer.")
    model = make_model1(input_shape=x_train.shape[1:])
    

model_summary_string = get_model_summary(model)
print(model_summary_string)


"""
## Train the model

"""
print("Training the model...")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        modelpath, save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    # patiencee set from 50 to 3 for experiement
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1),
]

opt = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    #validation_split=0.2,
    validation_data=(x_test, y_test),
    verbose=1,
)

"""
## Evaluate model on test data
"""

print("Evaluating the model...")

model = keras.models.load_model(modelpath)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print("LSTM layers   : ", layers)
print("Epoch limit   : ", epochs)
print("Batch size    : ", batch_size)
print("Test accuracy : ", test_acc)
print("Test loss     : ", test_loss)

"""
## Plot the model's training and validation loss
"""
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

metric = "loss"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

tf.keras.backend.clear_session()
