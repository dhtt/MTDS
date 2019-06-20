#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dohoangthutrang
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sb
import pandas as pd

cmap = sb.color_palette('plasma', 8)
#sb.palplot(cmap)

def evaluate_model(model, filename):
    model_history = model.history.history
    no_of_ran_epochs = len(model_history['loss'])
    plt.style.use('ggplot')
    plt.figure()
    
    #Plot model training loss 
    plt.subplot(211)
    plt.plot(np.arange(0, no_of_ran_epochs), model_history['loss'], label='Train loss')
    plt.plot(np.arange(0, no_of_ran_epochs), model_history['val_loss'], label='Val loss')
    plt.title('Training Loss and Accuracy of DNN model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.xlim([0.0, plt.xlim()[1]])
    plt.ylim([0.0, plt.ylim()[1]])
    
    #Plot model training accuracy
    plt.subplot(212)
    plt.plot(np.arange(0, no_of_ran_epochs), model_history['acc'], label='Train acc')
    plt.plot(np.arange(0, no_of_ran_epochs), model_history['val_acc'], label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(filename+'_Loss_n_Acc', dpi=300)
    plt.show()
    
def visualize_prediction(y, y_pred, filename):
    y_pred = pd.DataFrame(y_pred)
    y1 = y.values.flatten()
    y_pred1 = y_pred.values.flatten() 
    plt.style.use('ggplot')
    
    #Plot true vs predicted output 
    plt.figure()
    sb.scatterplot(y1, y_pred1, palette='plasma', hue=abs(y1-y_pred1), alpha=0.4)
    plt.title('True values vs Predicted values')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([plt.xlim()[0],plt.xlim()[1]])
    plt.ylim([plt.ylim()[0],plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.savefig(filename + '_True_vs_Fitted', figsize=(1,1), dpi=300)
    plt.show()
    plt.close()
    
    #Histogram of errors
    plt.figure()
    abs_error = y1-y_pred1
    sb.distplot(abs_error, color=cmap[1])
    plt.title('Error distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig(filename + '_Error_dist', figsize=(1,1), dpi=300)
    plt.show()
    plt.close()
    
    #Residual plot
    plt.figure()
    sb.jointplot(y_pred, y, kind='resid', scatter_kws={'alpha': 0.05}, color=cmap[0])
#    plt.title('Residuals Plot')
    plt.xlabel('Fitted value')
    plt.ylabel('Standardized Residual')
    plt.savefig(filename + '_Residual_plot', dpi=300)
    plt.show()
    plt.close()
  
def evaluate_binary_classification(y, y_pred, filename):
    y = y.values.flatten()
    y_pred = y_pred.flatten()
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    print(confusion_matrix)
    TP, TN, FP, FN = confusion_matrix[1, 1], confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0]
    classification_accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    sensitivity = TP/(FN+TP)
    specificity = TN/(TN+FP)
#    AUC = metrics.roc_auc_score(y, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    summary = pd.DataFrame([classification_accuracy,precision,sensitivity,specificity], index =['Classification Accuracy','Precision','Sensitivity','Specificity']).transpose()
    
    #Visualize confusion_matrix, classification_accuracy, precision, sensitivity, AUC
    plt.figure(1)
    sb.heatmap(confusion_matrix, annot=True, fmt=".0f", cbar=False, cmap=cmap, square=True, xticklabels=['Resistant (N)','Sensitive (N)'], yticklabels=['Resistant (T)','Sensitive (T)'])
    plt.title('Confusion Matrix')
    plt.savefig('Confusion_Matrix'+filename)
    plt.show()
    
    plt.figure(2)
    sb.barplot(data=summary, palette=cmap)
    plt.title('Prediction Strength')
    plt.savefig('Prediction_Strength'+filename)
    plt.show()
    
    plt.figure(3)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve binary classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.savefig('ROC_'+filename)
    plt.show()
    
def plot_grid_search(cv_score, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_score['mean_test_MSE']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_score['std_test_MSE']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

# Calling Method 
