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
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import KFold
import Assessment
from collections import defaultdict
import pickle
import seaborn as sb
import matplotlib.pyplot as plt

#Ignore warning messages while training
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def join_models(GE_dataset, CNV_dataset, MUT_dataset):
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
           
if __name__ == '__main__':
    tf.reset_default_graph()
    np.random.seed(0)
    batch_size = [5, 25, 50, 75, 100]
    epoch = 200
    cv_fold = 5
    GE = pd.read_csv('processed_GE.csv', delimiter=',', index_col = 0)
    CNV = pd.read_csv('processed_CNV.csv', delimiter=',', index_col = 0)
    MUT = pd.read_csv('processed_MUT.csv', delimiter=',', index_col = 0)
    Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0)
    model = join_models(GE, CNV, MUT)
    
    cv = KFold(n_splits=cv_fold, random_state=42, shuffle=True)
    test_loss_list_batch = defaultdict(list) #store test loss in dict whose keys are batchsizes
    val_loss_list_batch = defaultdict(list) #store val loss in dict whose keys are batchsizes
    testset_list_batch = defaultdict() #store y_test in dict whose keys are batchsizes for classfication
    pred_list_batch = defaultdict() #store y_pred in dict whose keys are batchsizes for classfication
    
    for i in batch_size:
        testset_list = defaultdict(list)
        pred_list = defaultdict(list)
        test_loss_list = [0]*epoch
        val_loss_list = [0]*epoch
        k=0 #to keep track of fold
        for train_index, test_index in cv.split(GE):
            logdir = "tf_logs/.../" + str(i)+ "/" + str(k) + "/"
            tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            print('Batch size: ',i, 'New fold: ', k)
            GE_train, GE_test = GE.iloc[train_index,], GE.iloc[test_index,]
            MUT_train, MUT_test = MUT.iloc[train_index,], MUT.iloc[test_index,]
            CNV_train, CNV_test = CNV.iloc[train_index,], CNV.iloc[test_index,]
            Y_train, Y_test = Y.iloc[train_index,], Y.iloc[test_index,]
            
            fitted_model = model.fit([GE_train, CNV_train, MUT_train], Y_train, 
               callbacks = [tensorboard], #can call early stopping here
               validation_data= [[GE_test, CNV_test, MUT_test], Y_test],
               epochs = epoch,
               batch_size = i)
            
            test_loss_list = [sum(n) for n in zip(test_loss_list, fitted_model.history['mean_squared_error'])] 
            val_loss_list = [sum(n) for n in zip(val_loss_list, fitted_model.history['val_mean_squared_error'])] 
            Y_pred = model.predict([GE_test, CNV_test, MUT_test])
            pred_list[k] = Y_pred
            testset_list[k] = Y_test
            k+=1
            
        testset_list_batch[i] = testset_list
        pred_list_batch[i] = pred_list
        test_loss_list_batch[i] = [x/cv_fold for x in test_loss_list] #the stored loss is averaged from all folds
        val_loss_list_batch[i] = [x/cv_fold for x in val_loss_list] #the stored loss is averaged from all folds
    
    '''Write output files'''
    with open('testset_list_batch.pickle', 'wb') as fout:
        pickle.dump(testset_list_batch, fout, protocol=pickle.HIGHEST_PROTOCOL)
#    with open('testset_list_batch.pickle', 'rb') as fin:
#        testset_list_batch_tmp = pickle.load(fin)
    
    with open('pred_list_batch.pickle', 'wb') as fout:
        pickle.dump(pred_list_batch, fout, protocol=pickle.HIGHEST_PROTOCOL)
#    with open('pred_list_batch.pickle', 'rb') as fin:
#        pred_list_batch = pickle.load(fin)

    with open('test_loss_list_batch.pickle', 'wb') as fout:
        pickle.dump(test_loss_list_batch, fout, protocol=pickle.HIGHEST_PROTOCOL)
#    with open('test_loss_list_batch.pickle', 'rb') as fin:
#        test_loss_list_batch = pickle.load(fin)
        
    with open('val_loss_list_batch.pickle', 'wb') as fout:
        pickle.dump(val_loss_list_batch, fout, protocol=pickle.HIGHEST_PROTOCOL)
#    with open('val_loss_list_batch.pickle', 'rb') as fin:
#        val_loss_list_batch = pickle.load(fin)
