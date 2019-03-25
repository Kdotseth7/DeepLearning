# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:02:22 2019

@author: Kushagra Seth
"""

# Artificial Neural Network (ANN)

# PART-1 : Data-Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])
oneHotEncoder = OneHotEncoder(categorical_features=[1]) # Dummy Variables
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) # No need to fit Test Set as its already fitted to Training Set

# PART-2 : Creating the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense 

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))

# Adding the output layer
classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

# Compiling ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Fitting ANN classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # To convert percentage into True or False for cm, equivalent of if condn

# Making the Confusion Matrix(Classification Evaluation Metric)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : %s " % (cm))

# =============================================================================
# # Pickling the Feature Scaler
# with open("sc.pickle", "wb") as s:
#     pickle.dump(sc, s)
#     
# # Pickling the classifier
# with open("classifier.pickle", "wb") as f:
#     pickle.dump(classifier, f)
# =============================================================================






  