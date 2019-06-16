#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:44:50 2019

@author: naajil
"""



import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import Assessment  

np.random.seed(7)
X1 = pd.read_csv('processed_GE.csv', delimiter=',', index_col = 0) #Investigate only 20 genes
#X2 = pd.read_csv('CNV_loss.csv', delimiter=',', index_col = 0).iloc[:30,:50] #Investigate only 20 genes
Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0) #Only 10 drugs/tasks

def build_DNN(input_height, input_width, output_width):
  model = keras.Sequential([
    Dense(256, activation='relu', input_shape=[input_width], name='Dense0'),
#    Dropout(0.5, name='Dropout0'),
    Dense(128, activation='relu', name='Dense1'),
#    Dropout(0.5, name='Dropout1'),
    Dense(128, activation='relu', name='Dense2'),
#    Dropout(0.5, name='Dropout2'),
    Dense(output_width, activation='linear', name='Dense3')
  ])
  model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy','mean_squared_error'])
  return model

def run_model(X, Y, no_of_epochs, batch_size, validation_split):
    #Define parameters
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    cvscores = []
    msescores = []
    x_height = X.shape[0]
    x_width = X.shape[1]
    y_width = Y.shape[1]
    #Build model
    model = build_DNN(x_height, x_width, y_width)
    model.summary()
    #Train model
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = X.iloc[train_index,], X.iloc[test_index,], Y.iloc[train_index,], Y.iloc[test_index,]
        model.fit(X_train, y_train, epochs = no_of_epochs, batch_size = batch_size)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f\n\n" % (model.metrics_names[2], scores[2]))
        cvscores.append(scores[1] * 100)
        msescores.append(scores[2])
    print("Overall Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("Overall MSE: %.2f (+/- %.2f)" % (np.mean(msescores), np.std(msescores)))
    #Make prediction
    y_pred = model.predict(X)
    #Evaluate model
    #Assessment.evaluate_model(no_of_epochs, model)
    #Assessment.visualize_prediction(Y, y_pred)
run_model(X1, Y, 10, 5, 0.25)