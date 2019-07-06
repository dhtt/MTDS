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
import Assessment

#Ignore warning messages while training
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_DNN(optimizer='Adam', dropout_rate=0.0, neurons1=128, neurons2=128,activation = 'relu'):
    #Create layers
    Inputs = Input(shape=[177], name='Inputs')
    x = Dense(neurons1, activation=activation)(Inputs)
    x = Dropout(dropout_rate, name='Dropout0')(x)
    x = Dense(neurons2, activation='relu')(x)
#    x = Dense(637, activation='relu')(x)
    Outputs = Dense(101, activation='linear', name='outputs')(x)
    #Compile model
    model = Model(inputs=Inputs, outputs=Outputs)
    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_squared_error', 'mean_absolute_error'])
    plot_model(model, show_shapes=True, to_file='model.png')
    return model

def optimize_hyperparameter(X, Y):
    #Define hypermeters values for optimization
    model = KerasRegressor(build_fn=build_DNN, verbose=0)
    batch_size = [5, 25, 50] #Add more later
    epochs = [250] #Add more later
    optimizer = ['RMSprop', 'Nadam'] #Reduce later
    drop_out = [0.1,0.2,0.3,0.4,0.5]
    activation = ['relu','sigmoid'] #reduce later
    neurons1 = [128*1, 128*2, 128*3, 128*4]
    neurons2 = [128*1, 128*2, 128*3, 128*4]
    param_grid = dict(neurons1=neurons1, neurons2=neurons2, 
                      optimizer=optimizer,
                      dropout_rate = drop_out,
                      activation = activation,
                      batch_size=batch_size, epochs=epochs)
    scoring = {'MSE':make_scorer(mean_squared_error, greater_is_better=False), 'MAE':make_scorer(mean_absolute_error, greater_is_better=False), 'EVS':make_scorer(explained_variance_score, greater_is_better=True)}
    #Run gridsearch
    grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                        cv=3, scoring=scoring, refit='MSE',n_jobs = -1) #Define grid with listed parameters. This includes inner CV and no. of folds is changed later to 10
    
    best_param_list = [] #list of best param combinations fromeach round of outer CV
    best_pred_list = [] #list of best predictions from each round of outer CV
    cv_score_list = [] #list of  cv scores of all models in all rounds of outer CV
    best_testset_list = []
    epoch_loss_list = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True) #Outer CV and no. of folds is changed later to 10
    for train_index, test_index in cv.split(X):
        print('New fold!')
        X_train, X_test, Y_train, Y_test = X.iloc[train_index,], X.iloc[test_index,], Y.iloc[train_index,], Y.iloc[test_index,]
        #X_train = pca.fit_transform(X_train)
        grid_result = grid.fit(X_train, Y_train) #Include inner cross-validation
        cv_score_list.append(grid_result.cv_results_)
        
        if grid_result.best_params_ not in best_param_list:
            best_param_list.append(grid_result.best_params_)
            print('New best param combi is added: ', grid_result.best_params_) #For checking
            
            #Predict X_test using newly obtained best model
            optimized_model = grid_result.best_estimator_ #Build the model using best params
            Y_pred = optimized_model.predict(X_test)
            best_pred_list.append(Y_pred)
            best_testset_list.append(Y_test)
            epoch_loss_list.append(grid_result.best_estimator_.model.history.history)
        else: #This is just for checking
            print('Current best combi is already there.')
    return(best_param_list, best_pred_list, cv_score_list, best_testset_list, epoch_loss_list)     

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
#    Assessment.evaluate_model(model, 'MUT')
    Assessment.visualize_prediction(Y, y_pred, 'MUT')
    return(y_pred)
#pred = run_model(X, Y, 10, 100, 0.2)

def main(agrv):
    if len(agrv) == 1:
        print('Ready to go!')
    else:
        print('No arguments is required.')
        sys.exit(2)
        
if __name__ == '__main__':
    #Check and configure GPUs usage 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    
    np.random.seed(0)
    
    #Training model
    X = pd.read_csv('processed_CNV.csv', delimiter=',', index_col = 0).iloc[:600,]
    Y = pd.read_csv('processed_DR.csv', delimiter=',', index_col = 0)
    optimization = optimize_hyperparameter(X,Y)
    
    #Get output files
    best_param_list = optimization[0]
    best_pred_list = optimization[1]
    cv_scores = optimization[2]
    best_testset_list = optimization[3]
    epoch_loss = optimization[4]
    
    cv_score_list=[]
    for i in cv_scores:
        j = { 'mean_fit_time': i['mean_fit_time'].tolist(), 'mean_score_time': i['mean_score_time'].tolist(),
             'mean_test_EVS': i['mean_test_EVS'].tolist(), 'mean_test_MAE': i['mean_test_MAE'].tolist(),
             'mean_test_MSE': i['mean_test_MSE'].tolist(), 'std_test_EVS': i['std_test_EVS'].tolist(), 
             'std_test_MAE': i['std_test_MAE'].tolist(), 'std_test_MSE': i['std_test_MSE'].tolist(),
             'std_fit_time': i['std_fit_time'].tolist(), 'std_score_time': i['std_score_time'].tolist()}
        cv_score_list.append(j)
    
    with open('best_param_list.csv', 'w') as fout:
        json.dump(best_param_list, fout)
    #with open('tempbest_param_list.csv') as fin:  
    #    data = json.load(fin)
        
    with open('cv_score_list.csv', 'w') as fout:
        json.dump(cv_score_list, fout)
    #with open('cv_score_list.csv') as fin:  
    #    data = json.load(fin)
    
    with open('best_pred_list.csv', 'w') as fout:
        for slice_2d in best_pred_list:
            np.savetxt(fout, slice_2d)
    #data=np.loadtxt('best_pred_list.csv')
    
    with open('params_list.csv', 'w') as fout:
        json.dump(cv_scores[0]['params'], fout)
#    with open('params_list.csv') as fin:  
#        data = json.load(fin)
        
    with open('best_testset_list.csv', 'w') as fout:
        for slice_2d in best_testset_list:
            np.savetxt(fout, slice_2d)
    #data=np.loadtxt('best_testset_list.csv')
    
    epoch_loss_list = []
    for i in epoch_loss:
        j = {'loss': str(i['loss']), 
             'mean_squared_error': str(i['mean_squared_error']),
             'mean_absolute_error': str(i['mean_absolute_error'])}
        epoch_loss_list.append(j)
        
    with open('epoch_loss_list.csv', 'w') as fout:
        json.dump(epoch_loss_list, fout)
#    with open('epoch_loss_list.csv') as fin:  
#        data = json.load(fin)
#    for i in data:
#        val = i['loss'][1:len(i['loss'])-1]
#        val_sp= val.split(',')
#        for j in val_sp:
#            print(j)
        
    Assessment.plot_score(cv_score_list, 'GE')
    print("--- %s seconds ---" % (time.time() - start_time))