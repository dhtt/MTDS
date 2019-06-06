#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dohoangthutrang
"""

import pandas as pd
import seaborn as sb
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, Flatten
import Assessment

np.random.seed(7)

def build_CNN1D_num(input_shape, output_width):
    model = keras.Sequential([
            Conv1D(filters=64, kernel_size=2, input_shape=input_shape[1:], activation='relu', name='Conv1D0'),
            Conv1D(filters=128, kernel_size=2, activation='relu', name='Conv1D1'),
            AveragePooling1D(name='MeanPooling0'),
            Conv1D(filters=128, kernel_size=2, activation='relu', name='Conv1D2'),
            Flatten(name='Flatten1'),
            Dense(128, activation='relu', name='Dense0'),
            Dense(64, activation='relu', name='Dense1'),
            Dense(output_width, activation='linear', name='Dense2')
            ])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy','mean_squared_error'])
    return model
#
def build_CNN1D_cat(input_shape, output_width):
    model = keras.Sequential([
            Conv1D(filters=64, kernel_size=2, input_shape=input_shape[1:], activation='relu', name='Conv1D0'),
            Conv1D(filters=128, kernel_size=2, activation='relu', name='Conv1D1'),
            AveragePooling1D(name='MeanPooling0'),
            Conv1D(filters=128, kernel_size=2, activation='relu', name='Conv1D2'),
            Flatten(name='Flatten1'),
            Dense(128, activation='relu', name='Dense0'),
            Dense(64, activation='relu', name='Dense1'),
            Dense(output_width, activation='sigmoid', name='Dense2')
            ])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy','mean_squared_error'])
    return model

def run_model(X, Y, model_type, no_of_epochs, batch_size, validation_split):
    X = np.asarray(X)
    X = X.reshape(X.shape+(1,))
    y_width = Y.shape[1]
    if model_type=='numerical': 
        #Build model
        model = build_CNN1D_num(X.shape, y_width)
        model.summary()
        #Train model
        model.fit(X, Y, epochs = no_of_epochs, batch_size = batch_size, validation_split = validation_split)
        y_pred = model.predict(X)
        #Evaluate model
        Assessment.evaluate_model(no_of_epochs, model)
        Assessment.visualize_prediction(Y, y_pred)
    elif model_type=='categorical':
        Y[Y>0.5] = 1
        Y[Y<0.5] = 0
        #Build model
        model = build_CNN1D_cat(X.shape, y_width)
        model.summary()
        #Train model
        model.fit(X, Y, epochs = no_of_epochs, batch_size=batch_size, validation_split = validation_split)
        y_pred = np.asarray(model.predict(X))
        y_pred[y_pred>0] = 1
        y_pred[y_pred<0] = 0
        #Evaluate model
        Assessment.evaluate_model(no_of_epochs, model)
        Assessment.visualize_prediction(Y, y_pred)
        Assessment.evaluate_binary_classification(Y, y_pred)


#X1 = pd.read_csv('ge_dataset.csv', delimiter=',', index_col = 0).iloc[:30,:30] #Investigate only 20 genes
X2 = pd.read_csv('CNV_dataset.csv', delimiter=',', index_col = 0).iloc[:30,] #Investigate only 20 genes
Y = pd.read_csv('drug_dataset.csv', delimiter=',', index_col = 0).iloc[:30,] #Only 10 drugs/tasks
sb.heatmap(X2)

run_model(X2, Y, 'categorical', 5, 1, 0.25)