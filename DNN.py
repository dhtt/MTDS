#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:34:47 2019

@author: dohoangthutrang
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  3 15:45:53 2019

@author: dohoangthutrang
'''

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import Assessment
from sklearn.metrics import make_scorer, mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold

np.random.seed(7)
X = pd.read_csv('processed_GE.csv', delimiter=',', index_col = 0)
Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0)

def build_DNN(optimizer='adam', activation='relu', dropout_rate=0.0, neurons=637):
    #Create layers
    Inputs = Input(shape=[637], name='Inputs')
    x = Dense(neurons, activation=activation)(Inputs)
#    x = Dropout(dropout_rate)
    x = Dense(637/5, activation='relu')(x)
#    x = Dense(637, activation='relu')(x)
    Outputs = Dense(101, activation='linear', name='outputs')(x)
    #Compile model
    model = Model(inputs=Inputs, outputs=Outputs)
    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_squared_error', 'mean_absolute_error'])
    plot_model(model, show_shapes=True, to_file='model.png')
    return model

def optimize_hyperparameter():
    #Define hypermeters values for optimization
    model = KerasRegressor(build_fn=build_DNN, verbose=0)
    batch_size = [100, 200] #Add more later
    epochs = [10] #Add more later
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] #Reduce later
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] #reduce later
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #Possibly change
    neurons = [1, 5, 10, 15, 20, 25, 30] #Change later
    param_grid = dict(batch_size=batch_size, epochs=epochs)
#                      , optimizer=optimizer, activation=activation, dropout_rate=dropout_rate, neurons=neurons)
    scoring = {'MSE':make_scorer(mean_squared_error), 'MAE':make_scorer(mean_absolute_error), 'EVS':make_scorer(explained_variance_score)}

    #Run gridsearch
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        cv=2, scoring=scoring, refit='MSE') #Define grid with listed parameters. This includes inner CV and no. of folds is changed later to 10
    
    best_param_list = [] #list of best param combinations from each round of outer CV
    best_pred_list = [] #list of best predictions from each round of outer CV
    cv_score_list = [] #list of  cv scores of all models in all rounds of outer CV
    
    cv = KFold(n_splits=2, random_state=42, shuffle=False) #Outer CV and no. of folds is changed later to 10
    for train_index, test_index in cv.split(X):
        X_train, X_test, Y_train, Y_test = X.iloc[train_index,], X.iloc[test_index,], Y.iloc[train_index,], Y.iloc[test_index,]
        grid_result = grid.fit(X_train, Y_train) #Include inner cross-validation
        cv_score_list.append(grid_result.cv_results_)
        
        if grid_result.best_params_ not in best_param_list:
            best_param_list.append(grid_result.best_params_)
            print('New best param combi is added!') #For checking
            
            #Predict X_test using newly obtained best model
            optimized_model = grid_result.best_estimator_ #Build the model using best params
            Y_pred = optimized_model.predict(X_test)
            Assessment.visualize_prediction(Y_test, Y_pred, str(grid_result.best_params_)) #Visualize predicted values and residuals
            best_pred_list.append(Y_pred)
        else: #This is just for checking
            print('Current best combi is already there.')
    return(best_param_list, best_pred_list, cv_score_list)     

optimization = optimize_hyperparameter()
best_params = optimization[0]
best_preds = optimization[1]
all_cv_scores = optimization[2]

#Assessment.plot_grid_search(cv_score, batch_size, epochs, 'Batch size', 'Epochs') #Supplement later

def run_model(X, Y, epochs, batch_size, validation_split):
    #Build model
    input_shape = X.shape[1]
    output_width = Y.shape[1]
    model = build_DNN()
    model.summary()
    
    #Train model
    ##Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, 
                                             save_weights_only=False, save_best_only=False, mode='max')
    
    model.fit(X, Y, 
              epochs = epochs, 
              batch_size = batch_size, 
              validation_split = validation_split,
              shuffle = True,
              callbacks=[])
    
    #Make prediction
    y_pred = pd.DataFrame(model.predict(X), columns=Y.columns, index=Y.index.values)
    
    #Evaluate model
    Assessment.evaluate_model(model, 'DNN1')
    Assessment.visualize_prediction(Y, y_pred, 'DNN1')
    return(y_pred)
#pred = run_model(X, Y, 10, 100, 0.2)

#def main(agrv):
#    if len(agrv) != 1:
#        print('Please check of the drug dataset has been given.')
#    else:
#        grid_result = optimize_hyperparameter()
#        pred = run_model(X, Y, 10, 100, 0.2)
#        print("--- %s seconds ---" % (time.time() - start_time))
#        sys.exit(2)
#        
#if __name__ == '__main__':
#    main(sys.argv[1:])