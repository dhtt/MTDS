#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:02:39 2019

@author: naajil
"""


import pandas as pd
import seaborn as sb
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, Flatten
from sklearn.model_selection import KFold
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
    #X = np.asarray(X)
    #X = X.reshape(X.shape+(1,))
    #X  = pd.DataFrame(list(map(np.ravel, X))).transpose()
    y_width = Y.shape[1]
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    cvscores = []
    msescores = []
    if model_type=='numerical': 
        #Build model
        #model = build_CNN1D_num(X.shape, y_width)
        #model.summary()
        #Train model
        for train_index, test_index in cv.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train_index,], X.iloc[test_index,], Y.iloc[train_index,], Y.iloc[test_index,]
            X_train = np.asarray(X_train)
            X_train = X_train.reshape(X_train.shape+(1,))
            
            X_test = np.asarray(X_test)
            X_test = X_test.reshape(X_test.shape+(1,))
            model = build_CNN1D_cat(X_train.shape, y_width)
            model.summary()
            model.fit(X_train, y_train, epochs = no_of_epochs, batch_size = batch_size)
            scores = model.evaluate(X_test, y_test, verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            print("%s: %.2f\n\n" % (model.metrics_names[2], scores[2]))
            cvscores.append(scores[1] * 100)
            msescores.append(scores[2])
        print("Overall Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        print("Overall MSE: %.2f (+/- %.2f)" % (np.mean(msescores), np.std(msescores)))
        #y_pred = model.predict(X)
        #Evaluate model
        #Assessment.evaluate_model(no_of_epochs, model)
        #Assessment.visualize_prediction(Y, y_pred)
    elif model_type=='categorical':
        Y[Y>0.5] = 1
        Y[Y<0.5] = 0
        #Build model
#        model = build_CNN1D_cat(X.shape, y_width)
#        model.summary()
        #Train model
        for train_index, test_index in cv.split(X):
            #print("TRainIndex: ",X[[687799]])
            
            X_train, X_test, y_train, y_test = X.iloc[train_index,], X.iloc[test_index,], Y.iloc[train_index,], Y.iloc[test_index,]
            X_train = np.asarray(X_train)
            X_train = X_train.reshape(X_train.shape+(1,))
            
            X_test = np.asarray(X_test)
            X_test = X_test.reshape(X_test.shape+(1,))
            model = build_CNN1D_cat(X_train.shape, y_width)
            model.summary()
            model.fit(X_train, y_train, epochs = no_of_epochs, batch_size=batch_size)
            scores = model.evaluate(X_test, y_test, verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            print("%s: %.2f\n\n" % (model.metrics_names[2], scores[2]))
            cvscores.append(scores[1] * 100)
            msescores.append(scores[2])
        print("Overall Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        print("Overall MSE: %.2f (+/- %.2f)" % (np.mean(msescores), np.std(msescores)))
        #y_pred = np.asarray(model.predict(X))
        #y_pred[y_pred>0.5] = 1
        #y_pred[y_pred<0.5] = 0
        #Evaluate model
        #Assessment.evaluate_model(no_of_epochs, model)
        #Assessment.visualize_prediction(Y, y_pred)
        #Assessment.evaluate_binary_classification(Y, y_pred)


#X1 = pd.read_csv('ge_dataset.csv', delimiter=',', index_col = 0).iloc[:30,:30] #Investigate only 20 genes
X2 = pd.read_csv('processed_CNV.csv', delimiter=',', index_col = 0).iloc[:600,] #Investigate only 20 genes
Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0).iloc[:600,] #Only 10 drugs/tasks


run_model(X2, Y, 'categorical', 10, 5, 0.25)