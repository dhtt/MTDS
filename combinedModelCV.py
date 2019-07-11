#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:30:48 2019

@author: naajil
"""

import time
start_time = time.time()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import make_scorer, mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.python.keras.layers.merge import concatenate
import Assessment

#Ignore warning messages while training
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def join_models(GE_dataset, CNV_dataset, MUT_dataset, optimizer = 'RMSprop'):
    shape_GE = GE_dataset.shape[1]
    shape_CNV = CNV_dataset.shape[1]
    shape_MUT = MUT_dataset.shape[1]
    
    Inputs_1 = Input(shape=[shape_GE], name='Inputs_1')
    x = Dense(384, activation='sigmoid', name='Dense_1_0')(Inputs_1)
    x = Dropout(0.2, name='Dropout_1_0')(x)
    x = Dense(512, activation='relu', name='Dense_1_1')(x)
    Outputs_1 = Dense(101, activation='linear', name='Outputs_1')(x)
    
    Inputs_2 = Input(shape=[shape_CNV], name='Inputs_2')
    y = Dense(256, activation='sigmoid', name='Dense_2_0')(Inputs_2)
    y = Dropout(0.5, name='Dropout_2_0')(y)
    y = Dense(256, activation='relu', name='Dense_2_1')(y)
    Outputs_2 = Dense(101, activation='sigmoid', name='Outputs_2')(y)
    
    Inputs_3 = Input(shape=[shape_MUT], name='Inputs_3')
    z = Dense(384, activation='relu', name='Dense_3_0')(Inputs_3)
    z = Dropout(0.4, name='Dropout_3_0')(z)
    z = Dense(512, activation='relu', name='Dense_3_1')(z)
    Outputs_3 = Dense(101, activation='linear', name='Outputs_3')(z)
    
    Concatenated = concatenate([Outputs_1, Outputs_2, Outputs_3], name='Concatenated')
    a = Dense(64, activation='relu', name='Dense_4_0')(Concatenated)
    a = Dense(64, activation='relu', name='Dense_4_1')(a)
    Main_output = Dense(101, activation='linear', name='Main_output')(a)
    
    model = Model(inputs=[Inputs_1, Inputs_2, Inputs_3], outputs=Main_output)
    model.compile(optimizer='RMSprop', loss='mean_squared_error',metrics=['mean_squared_error'])
    plot_model(model, show_shapes=True, to_file='join_models.png')
    return(model)

def optimize_hyperparameter(GE,MUT,CNV, Y, Number_of_epochs, Batch_Size):
    #Define hypermeters values for optimization
    msescores = []
    joined = join_models(GE, CNV, MUT)
    cv = KFold(n_splits=10, random_state=42, shuffle=True) #Outer CV and no. of folds is changed later to 10
    for train_index, test_index in cv.split(GE):
        print('New fold!')
        GE_train, GE_test = GE.iloc[train_index,], GE.iloc[test_index,]
        MUT_train, MUT_test = MUT.iloc[train_index,], MUT.iloc[test_index,]
        CNV_train, CNV_test = CNV.iloc[train_index,], CNV.iloc[test_index,]
        Y_train, Y_test = Y.iloc[train_index,], Y.iloc[test_index,]
        joined.fit([GE_train, CNV_train,MUT_train], Y_train, epochs = Number_of_epochs, batch_size = Batch_Size)
        scores = joined.evaluate([GE_test, CNV_test, MUT_test], Y_test, verbose=0)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("%s: %.2f\n\n" % (joined.metrics_names[1], scores[1]))
        msescores.append(scores[1])
        print("Overall MSE: %.2f (+/- %.2f)" % (np.mean(msescores), np.std(msescores)))
    return(np.mean(msescores),np.std(msescores))
    
    

def main(agrv):
    if len(agrv) == 1:
        print('Ready to go!')
    else:
        print('No arguments is required.')
        sys.exit(2)
        
if __name__ == '__main__':
    np.random.seed(0)
    MSE_Score_List = []
    batch_size = [5,25,50] #Add more later
    epochs = [25,50,75,100,125,150,175,200,225,250] #Add more later
    GE_dataset = pd.read_csv('processed_GE.csv', delimiter=',', index_col = 0).iloc[:500,:]
    CNV_dataset = pd.read_csv('processed_CNV.csv', delimiter=',', index_col = 0).iloc[:500,:]
    MUT_dataset = pd.read_csv('processed_MUT.csv', delimiter=',', index_col = 0).iloc[:500,:]
    Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0).iloc[:500,:]
    #Training model
    for i in batch_size:
        for j in epochs: 
            optimization = optimize_hyperparameter(GE_dataset,CNV_dataset,MUT_dataset,Y,j,i)
            MSE_Mean = optimization[0]
            MSE_SD = optimization[1]
            print('MSE Mean = ' , MSE_Mean,'Standard Deviation: ', MSE_SD )
            temp={"Overall_MSE":MSE_Mean,"Standard_Deviation":MSE_SD,"Number_of_Epochs": j,"Batch_Size":i}
            MSE_Score_List.append(temp)

    
    with open('MSE_Scores.csv', 'w') as fout:
        json.dump(str(MSE_Score_List), fout)
        
    print("--- %s seconds ---" % (time.time() - start_time))
