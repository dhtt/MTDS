#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dohoangthutrang
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def evaluate_model(number_of_epochs, model):
    model_history = model.history.history
    plt.style.use('ggplot')
    plt.figure()
    plt.subplot(211)
    plt.plot(np.arange(0, number_of_epochs), model_history['loss'], label='Train loss')
    plt.plot(np.arange(0, number_of_epochs), model_history['val_loss'], label='Val loss')
    plt.title('Training Loss and Accuracy of DNN model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.xlim([0.0, plt.xlim()[1]])
    plt.ylim([0.0, plt.ylim()[1]])
    plt.subplot(212)
    plt.plot(np.arange(0, number_of_epochs), model_history['acc'], label='Train acc')
    plt.plot(np.arange(0, number_of_epochs), model_history['val_acc'], label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    plt.close()
    
def visualize_prediction(y, y_pred):
    #PLot y versus y_pred
    y = y.values.flatten()
    y_pred = y_pred.flatten()
    plt.style.use('ggplot')
    plt.figure()
    plt.subplot(121)
    plt.scatter(y, y_pred)
    plt.title('True values vs Predicted values ')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([plt.xlim()[0],plt.xlim()[1]])
    plt.ylim([plt.ylim()[0],plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    #Histogram of errors
    plt.subplot(122)
    abs_error = y-y_pred
    plt.hist(abs_error)
    plt.title('Error distribution')
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.show()
    plt.close()

def evaluate_binary_classification(y, y_pred):
    y = y.values.flatten()
    y_pred = y_pred.flatten()
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    print(confusion_matrix)
    TP, TN, FP, FN = confusion_matrix[1, 1], confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0]
    classification_accuracy = (FP+FN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    sensitivity = TP/(FN+TP)
    specificity = TN/(TN+FP)
    AUC = metrics.roc_auc_score(y, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve binary classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)