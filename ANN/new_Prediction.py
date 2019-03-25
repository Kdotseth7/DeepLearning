# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:03:14 2019

@author: Kushagra Seth
"""
# Importing the libraries
import numpy as np
import pickle

# Unpickling the Feature Scaler
with open("sc.pickle", "rb") as s:
    sc = pickle.load(s)
    
# Unpickling the classifier
with open("classifier.pickle", "rb") as f:
    clf = pickle.load(f)

# Predicting a single observation
"""
Predict if the customer with following information will leave the bank :
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40
    Tenure: 3
    Balance: 60000
    Number of Products: 2
    Has Credit Card: Yes
    Is Active Member: Yes
    Estimated Salary: 50000
"""

new_Prediction = clf.predict(sc.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_Prediction = (new_Prediction > 0.5)