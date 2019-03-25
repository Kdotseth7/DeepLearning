# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:46:33 2019

@author: Kushagra Seth
"""

# Convolutional Neural Network (CNN)

# PART-1 : Creating the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step-1: Convolution
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation="relu"))

# Step-2: Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding second convolution layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step-3: Flattening
classifier.add(Flatten())

#Step-4: Full Connection
classifier.add(Dense(output_dim=128, activation="relu"))
classifier.add(Dense(output_dim=1, activation="sigmoid")) # sigmoid because o/p is binary otherwise use softmax

# Compiling CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # adam - stochastic gradient
                                                                                       # loss = "binary_crossentropy" bcoz o/p is binary


# Part-2 : Fitting CNN classifier to the Training set     
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')      

import scipy.ndimage
classifier.fit_generator( training_set,
        steps_per_epoch=8000/32,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/32)   
                                                                                           