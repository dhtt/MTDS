#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 01:40:22 2019

@author: dohoangthutrang
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import make_scorer, mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
import Assessment
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#Ignore warning messages while training

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def join_models(GE_dataset, CNV_dataset, MUT_dataset):
    shape_GE = GE_dataset.shape[1]
    shape_CNV = CNV_dataset.shape[1]
    shape_MUT = MUT_dataset.shape[1]
    
    Inputs_1 = Input(shape=[shape_GE], name='Inputs_1')
    x = Dense(128, activation='relu', name='Dense_1_0')(Inputs_1)
    x = Dense(128, activation='relu', name='Dense_1_1')(x)
    Outputs_1 = Dense(101, activation='linear', name='Outputs_1')(x)
    
    Inputs_2 = Input(shape=[shape_CNV], name='Inputs_2')
    y = Dense(128, activation='relu', name='Dense_2_0')(Inputs_2)
    y = Dense(128, activation='relu', name='Dense_2_1')(y)
    Outputs_2 = Dense(101, activation='sigmoid', name='Outputs_2')(y)
    
    Inputs_3 = Input(shape=[shape_MUT], name='Inputs_3')
    z = Dense(128, activation='relu', name='Dense_3_0')(Inputs_3)
    z = Dense(256, activation='relu', name='Dense_3_1')(z)
    Outputs_3 = Dense(101, activation='linear', name='Outputs_3')(z)
    
    Concatenated = concatenate([Outputs_1, Outputs_2, Outputs_3], name='Concatenated')
    a = Dense(64, activation='relu', name='Dense_4_0')(Concatenated)
    a = Dense(64, activation='relu', name='Dense_4_1')(a)
    Main_output = Dense(101, activation='linear', name='Main_output')(a)
    
    model = Model(inputs=[Inputs_1, Inputs_2, Inputs_3], outputs=Main_output)
    model.compile(optimizer='Adam', loss='mean_squared_error')
    plot_model(model, show_shapes=True, to_file='join_models.png')
    return(model)
    
if __name__ == '__main__':
    np.random.seed(0)
    
    GE_dataset = pd.read_csv('processed_GE.csv', delimiter=',', index_col = 0).iloc[:500,:]
    CNV_dataset = pd.read_csv('processed_CNV.csv', delimiter=',', index_col = 0).iloc[:500,:]
    MUT_dataset = pd.read_csv('processed_MUT.csv', delimiter=',', index_col = 0).iloc[:500,:]
    Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0).iloc[:500,:]
    
    joined = join_models(GE_dataset, CNV_dataset, MUT_dataset)
    joined.fit([GE_dataset, CNV_dataset, MUT_dataset], 
               Y, 
               epochs = 50, 
               batch_size = 50)