#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  3 15:45:53 2019

@author: dohoangthutrang
'''


import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import Assessment  

np.random.seed(7)
X1 = pd.read_csv('ge_dataset.csv', delimiter=',', index_col = 0).iloc[:30,] #Investigate only 20 genes
#X2 = pd.read_csv('CNV_loss.csv', delimiter=',', index_col = 0).iloc[:30,:50] #Investigate only 20 genes
Y = pd.read_csv('drug_dataset.csv', delimiter=',', index_col = 0).iloc[:30,] #Only 10 drugs/tasks

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
    x_height = X.shape[0]
    x_width = X.shape[1]
    y_width = Y.shape[1]
    #Build model
    model = build_DNN(x_height, x_width, y_width)
    model.summary()
    #Train model
    model.fit(X, Y, epochs = no_of_epochs, batch_size = batch_size, validation_split = validation_split)
    #Make prediction
    y_pred = model.predict(X)
    #Evaluate model
    Assessment.evaluate_model(no_of_epochs, model)
    Assessment.visualize_prediction(Y, y_pred)
run_model(X1, Y, 2, 5, 0.25)