#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:32:55 2019

@author: dohoangthutrang
"""

'''
VISUALIZATION OF SEPARATE MODEL
'''
'''Import all output files there are'''
import json
import numpy as np
import pandas as pd
import Assessment
import seaborn as sb
import matplotlib.pyplot as plt
from collections import defaultdict
sb.set_style("whitegrid")
cmap = sb.color_palette('plasma', 8)

with open('GE_best_param_list.csv') as fin: 
    best_param_list = json.load(fin)
with open('GE_params_list.csv') as fin: 
    params_list = json.load(fin)
with open('GE_cv_score_list.csv') as fin:  
    all_cv_scores = json.load(fin)

pred_list=pd.DataFrame(np.loadtxt('MUT_best_pred_list.csv'))
test_list=pd.DataFrame(np.loadtxt('MUT_best_testset_list.csv'))

with open('GE_epoch_loss_list.csv') as fin:  
    epoch_loss_list = json.load(fin)
    
'''Binary classification'''
thres = pd.read_csv('threshold.csv', delimiter=';', index_col = 0, header=0, error_bad_lines=False).to_dict()
thres = thres['Threshold (logIC50)']
thres['Mitomycin-C'] = thres.pop('Mitomycin C')
thres['BAY-61-3606'] = thres.pop('BAY 61-3606')
thres['GSK1904529A'] = thres.pop('GSK-1904529A')
Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0)

def drop_new_drugs(pred_list, Y):
    pred_list_new = pred_list.copy()
    pred_list_new.columns = Y.columns
    drop_list = []
    for i in pred_list_new.columns:
        if i not in thres.keys():
            drop_list.append(i)
    pred_list_new = pred_list_new.drop(drop_list, axis=1)
    return(pred_list_new)
    
def binarize_output(Y_new):    
    store_2=[]
    for i in Y_new.columns:
        store_1 = []
        for j in Y_new[i]:
            if j>thres[i]:
                store_1.append(0) #Resistance = 0
            else:
                store_1.append(1) #Sensitive = 0
        store_2.append(store_1)
    binarized_output = pd.DataFrame(store_2).transpose()
    binarized_output.columns = Y_new.columns
    binarized_output.index=list(Y_new.index.values)
    return(binarized_output)
    
pred_list_new = drop_new_drugs(pred_list, Y)
test_list_new = drop_new_drugs(test_list, Y)
pred_list_new = binarize_output(pred_list_new)
test_list_new = binarize_output(test_list_new)

Assessment.evaluate_binary_classification(test_list_new, pred_list_new,'MUT')

'''Plot best models over 200 epochs'''
def plot_best_models(epoch_loss_list, filename):
    models_MSE = []
    for k in epoch_loss_list:
        l = k['loss'][1:len(k['loss'])-1]
        l= l.split(',')
        MSE=[]
        for i in l:
            MSE.append(float(i))
        models_MSE.append(MSE)
    epoch = list(range(1,201))
    models_MSE.append(epoch)
    models_MSE = pd.DataFrame(models_MSE).transpose()
    models_MSE['Type'] = 'Train loss'
    models_MSE.columns = ['Model 1','Model 2','Model 3','Model 4','Model 5','Epoch','Type']
    score_df_melt = pd.melt(models_MSE, value_vars=['Model 1','Model 2','Model 3','Model 4','Model 5'],
                       value_name='MSE', id_vars=['Type','Epoch'], var_name='Model')
    plt.figure()
    sb.lineplot(x='Epoch', y='MSE', data=score_df_melt,
              hue='Type', palette=sb.color_palette('plasma', 1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Loss of Best models (MSE)', fontsize=18)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.axhline(y=0, color='r')
    plt.savefig(filename + '_Best_models',  dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
plot_best_models(epoch_loss_list, 'GE')

'''Plot models fit time and MSE for all grid params'''
optimizer, neurons2, neurons1, batch_size, activator, dropout = [],[],[],[],[],[]
index = list(range(0,960))
for i in params_list:
    optimizer.append(i['optimizer'])
    neurons2.append(i['neurons2'])
    neurons1.append(i['neurons1'])
    batch_size.append(i['batch_size'])
    activator.append(i['activation'])
    dropout.append(i['dropout_rate'])
models = [index, optimizer, neurons2, neurons1, dropout, batch_size, activator]
models = pd.DataFrame(models).transpose()
models.columns = ['ID','Optimizer','Neurons 2','Neurons 1','Dropout rate', 'Batch size','Activator']

i=0
MSE, MAE, EVS, fit_time = [],[],[],[]
for i in range(len(all_cv_scores)):
    MSE.append(all_cv_scores[i]['mean_test_MSE'])
    MAE.append(all_cv_scores[i]['mean_test_MAE'])
    EVS.append(all_cv_scores[i]['mean_test_EVS'])
    fit_time.append(all_cv_scores[i]['mean_fit_time'])
MSE = -pd.DataFrame(MSE).transpose()
MAE = -pd.DataFrame(MAE).transpose()
EVS = pd.DataFrame(EVS).transpose()
fit_time = pd.DataFrame(fit_time).transpose()

def model_list(parameter):
    store_1 = defaultdict()
    m = []
    for i in range(486):
        if parameter == 'Optimizer':
            id = [models['Neurons 2'][i],models['Neurons 1'][i],models['Dropout rate'][i],models['Batch size'][i],models['Activator'][i]]
        elif parameter == 'Neurons 2':
            id = [models['Optimizer'][i],models['Neurons 1'][i],models['Dropout rate'][i],models['Batch size'][i],models['Activator'][i]]
        elif parameter == 'Neurons 1':
            id = [models['Optimizer'][i],models['Neurons 2'][i],models['Dropout rate'][i],models['Batch size'][i],models['Activator'][i]]
        elif parameter == 'Dropout rate':
            id = [models['Optimizer'][i],models['Neurons 2'][i],models['Neurons 1'][i],models['Batch size'][i],models['Activator'][i]]
        elif parameter == 'Batch size':
            id = [models['Optimizer'][i],models['Neurons 2'][i],models['Neurons 1'][i],models['Dropout rate'][i],models['Activator'][i]]
        elif parameter == 'Activator':
            id = [models['Optimizer'][i],models['Neurons 2'][i],models['Neurons 1'][i],models['Dropout rate'][i],models['Batch size'][i]]       
        id=str(id)
        if id not in store_1.keys():
            store_1[id] = i
            m.append(i)
        else:
            m.append(store_1[id])
    m = pd.DataFrame(m)
    return(m)

def plot_time(models, filename):
    temp = models.copy() 
    temp['op'] = model_list('Optimizer')
    temp['n2'] = model_list('Neurons 2')
    temp['n1'] = model_list('Neurons 1')
    temp['do'] = model_list('Dropout rate')
    temp['bs'] = model_list('Batch size')
    temp['ac'] = model_list('Activator')
    temp['Fold 1'] = fit_time[0]
    temp['Fold 2'] = fit_time[1]
    temp['Fold 3'] = fit_time[2]
    temp['Fold 4'] = fit_time[3]
    temp['Fold 5'] = fit_time[4]
    score_df_melt = pd.melt(temp, value_vars=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'], 
                  id_vars=['Optimizer','Neurons 2','Neurons 1', 'Dropout rate', 'Batch size','Activator'],
                  var_name='Fold', value_name='Fit time')
    
    f1 = sb.relplot(x="Dropout rate", y="Fit time", col="Batch size", data=score_df_melt,
           style="Optimizer", hue='Activator', palette=sb.color_palette('plasma', 2),
           kind='line', height=3, aspect=1,)
    f1.fig.subplots_adjust(top=0.7)
    f1.fig.suptitle('Mean Fit Time (s)',fontsize=20)
    for ax in f1.axes.flat:
        ax.axhline(y=0, color='r')
    plt.savefig(filename+'_CV_Fittime1', dpi=300)
    plt.show()
    plt.close()
    f2 = sb.relplot(x="Dropout rate", y="Fit time", col='Neurons 2', row='Neurons 1', data=score_df_melt,
           style="Optimizer", hue='Activator', palette=sb.color_palette('plasma', 2),
           kind='line', height=3, aspect=1,)
    f2.fig.subplots_adjust(top=0.9)
    f2.fig.suptitle('Mean Fit Time (s)',fontsize=22)
    for ax in f2.axes.flat:
        ax.axhline(y=0, color='r')
    plt.savefig(filename+'_CV_Fittime2', dpi=300)
    plt.show()
    plt.close()
#plot_time(models, 'GE')

def plot_MSE(models, filename):
    temp = models.copy() 
    temp['op'] = model_list('Optimizer')
    temp['n2'] = model_list('Neurons 2')
    temp['n1'] = model_list('Neurons 1')
    temp['do'] = model_list('Dropout rate')
    temp['bs'] = model_list('Batch size')
    temp['ac'] = model_list('Activator')
    temp['Fold 1'] = MSE[0]
    temp['Fold 2'] = MSE[1]
    temp['Fold 3'] = MSE[2]
    temp['Fold 4'] = MSE[3]
    temp['Fold 5'] = MSE[4]
    score_df_melt = pd.melt(temp, value_vars=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'], 
                  id_vars=['Optimizer','Neurons 2','Neurons 1', 'Dropout rate', 'Batch size','Activator'],
                  var_name='Fold', value_name='MSE')
    
    f1 = sb.relplot(x="Dropout rate", y="MSE", col="Batch size", data=score_df_melt,
           style="Optimizer", hue='Activator', palette=sb.color_palette('plasma', 2),
           kind='line', height=3, aspect=1,)
    f1.fig.subplots_adjust(top=0.7)
    f1.fig.suptitle('Cross Validation Score (MSE)',fontsize=20)
    for ax in f1.axes.flat:
        ax.axhline(y=0, color='r')
    plt.savefig(filename+'_CV_MSE1', dpi=300)
    plt.show()
    plt.close()
    
    f2 = sb.relplot(x="Dropout rate", y="MSE", col='Neurons 2', row='Neurons 1', data=score_df_melt,
           style="Optimizer", hue='Activator', palette=sb.color_palette('plasma', 2),
           kind='line', height=3, aspect=1,)
    f2.fig.subplots_adjust(top=0.9)
    f2.fig.suptitle('Cross Validation Score (MSE)',fontsize=22)
    for ax in f2.axes.flat:
        ax.axhline(y=0, color='r')
    plt.savefig(filename+'_CV_MSE2', dpi=300)
    plt.show()
    plt.close()
plot_MSE(models, 'GE')

'''
VISUALIZATION OF JOINED MODEL
'''

'''Import all output files there are'''
import pickle
with open('testset_list_batch.pickle', 'rb') as fin:
    testset_list_batch = pickle.load(fin)

with open('pred_list_batch.pickle', 'rb') as fin:
    pred_list_batch = pickle.load(fin)
    
with open('test_loss_list_batch.pickle', 'rb') as fin:
    test_loss_list_batch = pickle.load(fin)
    
with open('val_loss_list_batch.pickle', 'rb') as fin:
    val_loss_list_batch = pickle.load(fin)

'''Plot test loss and val loss'''
def plot_train_and_val_MSE(test_loss_list_batch,val_loss_list_batch):
    
    test_loss_list_batch = pd.DataFrame(test_loss_list_batch)
    test_loss_list_batch['Epoch'] = list(range(1,201))
    test_loss_list_batch['Type'] = 'Train loss'
    
    val_loss_list_batch = pd.DataFrame(val_loss_list_batch)
    val_loss_list_batch['Epoch'] = list(range(1,201))
    val_loss_list_batch['Type'] = 'Val loss'
    
    loss = pd.concat([test_loss_list_batch,val_loss_list_batch])
    loss = pd.melt(loss,id_vars=['Type','Epoch'], var_name='Batch size', value_name='MSE')
    plt.figure()
    sb.lineplot(x='Epoch', y='MSE', data=loss,
                hue='Batch size', style='Type', dashes=['',(2, 1)],
                palette=sb.color_palette('plasma', 5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Cross Validation Score (MSE)', fontsize=18)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.axhline(y=0,color='r')
    plt.savefig('join_CV_MSE', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 
plot_train_and_val_MSE(test_loss_list_batch,val_loss_list_batch)

def get_ypred_or_ytest(batchsize, option):
    y_list = []
    if option =='pred':
        for i in pred_list_batch[batchsize]:
            k = pred_list_batch[batchsize][i]
            for j in k:
                y_list.append(j)
    elif option =='test':
        for i in testset_list_batch[batchsize]:
            k = testset_list_batch[batchsize][i].to_numpy()
            for j in k:
                y_list.append(j)
    y_list = pd.DataFrame(y_list)
    return(y_list)

'''Plot histogram of error'''
batchsize = [5,25, 50, 75, 100]
cmap = sb.color_palette('plasma', 8)
i=0
plt.figure()
for bs in batchsize: 
    y_pred = get_ypred_or_ytest(bs, 'pred')
    y_test = get_ypred_or_ytest(bs, 'test')
    
    y_test_h = y_test.values.flatten()
    y_pred_h = y_pred.values.flatten() 
    filename = 'joinmodel'+str(bs)
    abs_error = y_test_h-y_pred_h
    sb.distplot(abs_error, color = cmap[i], 
                hist_kws=dict(alpha=0.1),
                kde_kws={"lw":1.5, "label": str(bs)},)
    i+=1
plt.title('Error distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.ylim(0,1)
plt.savefig('joinmodel_Error_dist', figsize=(1,1), dpi=300)
plt.show()
plt.close()


'''Binary classification'''
for bs in batchsize: 
    y_pred = get_ypred_or_ytest(bs, 'pred')
    y_test = get_ypred_or_ytest(bs, 'test')
   
    pred_list_new = drop_new_drugs(y_pred, Y)
    test_list_new = drop_new_drugs(y_test, Y)
    pred_list_new = binarize_output(pred_list_new)
    test_list_new = binarize_output(test_list_new)
    
    filename = 'joinmodel'+str(bs)
    Assessment.evaluate_binary_classification(pred_list_new, test_list_new,filename)
