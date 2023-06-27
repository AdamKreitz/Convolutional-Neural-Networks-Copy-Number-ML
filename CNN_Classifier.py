## All Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
import keras
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
## Define Functions

## Function for plotting ROC Curves
def roc_plot(predictions, truths, dataset):
    fpr, tpr, _ = roc_curve(truths, predictions)
    roc_auc = roc_auc_score(truths, predictions)
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='indigo', lw=2, label='ROC curve (AUC = {0:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    #plt.axvline(x=0.02)
    #plt.axvline(x=0.05)
    #plt.axhline(y=0.3)
    #plt.axhline(y=0.43)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve at TF >= 0.05')
    plt.legend(loc="lower right")
    plt.savefig('plots/' + dataset)
    #plt.close()
    
    
## Model built specifically for instantiating a CNN for training on the Delfi Dataset
def save_model_activity_delfi(images, labels):
        # initial training with validation hold-out
        verbose, epochs, batch_size = 1, 40, 60
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        fold = 1
        predictions = []
        ground_truth = []
        res = []
        truth = []
        for _ in range(50):
            for train_index, test_index in skf.split(images, labels):
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
                print('#################### FOLD ' + str(fold) + ' ####################')
                train_index = train_index.astype(int)
                images_train = images[train_index]
                labels_train = labels[train_index]  ###########################################################
                images_test = images[test_index]
                labels_test = labels[test_index]  #############################################################
                steps, features, outputs = images_train.shape[1], images_train.shape[2], 1
                print('#################### TRAINING ####################')

                model = Sequential()

                model.add(Conv1D(filters= 90, kernel_size=4, padding='same', activation='relu', input_shape=(images_train.shape[1], 1)))

                model.add(MaxPooling1D(pool_size=2))

                model.add(Dropout(0.15))

                model.add(Conv1D(filters=30, kernel_size=2, padding='same', activation='relu'))

                model.add(MaxPooling1D(pool_size=1))

                model.add(Dropout(0.15))

                model.add(Flatten())

                model.add(Dense(130, activation='relu'))

                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])



                history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                    shuffle=True, validation_data=(images_test, labels_test))
                print('#################### TESTING ####################')
                predictions.append(model.predict(images_test))
                ground_truth.append(labels_test)
                iter_res = []
                iter_truth = []
                for i in predictions:
                    for j in i:
                        res.append(j[0])
                        iter_res.append(j[0])
                labels_test = np.concatenate(ground_truth)
                for i in (labels_test):
                    truth.append(i)
                    iter_truth.append(i)
                print(roc_auc_score(iter_truth, iter_res))
                fold += 1
            #predictions = np.concatenate(predictions)
            #for i in predictions:
              #  res.append(i[0])
            #labels_test = np.concatenate(ground_truth)
            #for i in range(len(labels_test)):
                #for j in labels_test[i]:
             #       truth.append(i)
            #vals = model.predict(val_i)
            #val_res = []
            #for i in vals:
            #    val_res.append(i[0])
            #print(roc_plot(np.array(val_res), np.array(val_l)))
            #print ROC and AUC from the K-fold validation
        
        return roc_plot(np.array(iter_res), np.array(iter_truth), 'delfi'), iter_res, iter_truth

## Model built specifically for instantiating a CNN for training on the Delfi Dataset    
def save_model_activity_lucas(images, labels):
        verbose, epochs, batch_size = 1, 30, 120
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        fold = 1
        predictions = []
        ground_truth = []
        res = []
        truth = []
        for _ in range(10):
            for train_index, test_index in skf.split(images, labels):
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
                print('#################### FOLD ' + str(fold) + ' ####################')
                train_index = train_index.astype(int)
                images_train = images[train_index]
                labels_train = labels[train_index]  ###########################################################
                images_test = images[test_index]
                labels_test = labels[test_index]  #############################################################
                steps, features, outputs = images_train.shape[1], images_train.shape[2], 1
                print('#################### TRAINING ####################')

                model = Sequential()

                model.add(Conv1D(filters= 120, kernel_size=4, padding='same', activation='relu', input_shape=(images_train.shape[1], 1)))

                model.add(MaxPooling1D(pool_size=2))

                model.add(Dropout(0.25))

                model.add(Conv1D(filters=20, kernel_size=2, padding='same', activation='relu'))

                model.add(MaxPooling1D(pool_size=1))

                model.add(Dropout(0.2))

                model.add(Flatten())

                model.add(Dense(120, activation='relu'))

                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])



                history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                    shuffle=True, validation_data=(images_test, labels_test))
                print('#################### TESTING ####################')
                predictions.append(model.predict(images_test))
                ground_truth.append(labels_test)
                iter_res = []
                iter_truth = []
                for i in predictions:
                    for j in i:
                        res.append(j[0])
                        iter_res.append(j[0])
                labels_test = np.concatenate(ground_truth)
                for i in (labels_test):
                    truth.append(i)
                    iter_truth.append(i)
                print(roc_auc_score(iter_truth, iter_res))
                fold += 1
            #predictions = np.concatenate(predictions)
            #for i in predictions:
              #  res.append(i[0])
            #labels_test = np.concatenate(ground_truth)
            #for i in range(len(labels_test)):
                #for j in labels_test[i]:
             #       truth.append(i)
            #vals = model.predict(val_i)
            #val_res = []
            #for i in vals:
            #    val_res.append(i[0])
            #print(roc_plot(np.array(val_res), np.array(val_l)))
            #print ROC and AUC from the K-fold validation
        
        return roc_plot(np.array(iter_res), np.array(iter_truth), 'lucas'), iter_res, iter_truth

## Function for instantiating a basic CNN model
def make_model():

    model = Sequential()

    model.add(Conv1D(filters= 160, kernel_size=4, padding='same', activation='relu', input_shape=(val_df.shape[1], 1)))

    model.add(MaxPooling1D(pool_size=3))

    model.add(Dropout(0.15))

    model.add(Conv1D(filters=30, kernel_size=2, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=1))

    model.add(Dropout(0.15))

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
    
    return model
    
    
## Load input dataframes

# Delfi Cohort

delfi_df = pd.read_csv('data/delfi_df', index_col=0)

delfi_df = delfi_df.apply(pd.to_numeric, errors='coerce')

delfi_labels = pd.read_csv('delfi_labels', index_col=0)

# Lucas, Validation Cohorts

lucas_df = pd.read_csv('data/lucas_df', index_col=0)
validation_df = pd.read_csv('data/validation_df', index_col=0)

lucas_labels = pd.read_csv('data/lucas_labels', index_col=0)
validation_labels = pd.read_csv('data/validation_labels', index_col=0)


## Reshape Input Matrices-Delfi

delfi_arr = np.array(delfi_df).reshape((delfi_df.shape[0], delfi_df.shape[1], 1))
delfi_labels_arr = np.array(delfi_labels)
#save_model_activity_delfi(delfi_arr, delfi_labels_arr)

## Reshape Input Matrices-Lucas

lucas_arr = np.array(lucas_df).reshape((lucas_df.shape[0], lucas_df.shape[1], 1))
lucas_labels_arr = np.array(lucas_labels)

save_model_activity_lucas(lucas_arr, lucas_labels_arr)
    
    
## Validation

validation_arr = np.array(validation_df).reshape((validation_df.shape[0], validation_df.shape[1], 1))

res = []
lab = []
for i in range(50):
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
    model = make_model()
    model.fit(lucas_arr, lucas_labels_arr, epochs = 30, batch_size = 125, validation_data = (validation_arr, validation_labels))
    res.append(model.predict(val_arr))
    lab.append(val_cancer)
    #print(roc_plot(model.predict(val_arr), val_cancer))    

fin = []
for i in range(len(res)):
    #print(res[i])
    for j in range(len(res[i])):
        fin.append(res[i][j][0])    
    
labs = []
for i in np.array(lab):
    for j in i:
        labs.append(j[0])    
    
roc_plot(fin, labs, 'validation')    
    
    
    
    
    
    
    
    
    
