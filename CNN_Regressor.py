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
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

## Define Functions

def roc_plot(predictions, truths, name):
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
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/' + name)
    #plt.close()



def make_model():

    model = Sequential()

    model.add(Conv1D(filters = 28, kernel_size=6, padding='same', activation='relu', input_shape=(5006, 1)))

    model.add(MaxPooling1D(pool_size=3))

    model.add(Dropout(0.15))

    model.add(Conv1D(filters=8, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=1))

    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='MeanAbsoluteError', optimizer='adam', metrics=['MeanAbsoluteError'])
    
    return model

    
    
## Load Relevant DataFrames

## Dataset containing CNA data on mCRPC admixtures
crpc_df = pd.read_csv('data/crpc_df', index_col=0)
## Dataset containing CNA data on LuCaP admixtures
lucap_df = pd.read_csv('data/lucap_df', index_col=0)
## Data set containing Tumor fraction labels
tumor_fraction = pd.read_csv('data/tumor_fraction', index_col=0)


## Deep Learning

## Reshape input Matrix

train_arr = np.array(lucap_df).reshape(lucap_df.shape[0], lucap_df.shape[1], 1)
valid_arr = np.array(crpc_df).reshape(crpc_df.shape[0], crpc_df.shape[1], 1)

## Run X iterations while testing on the validation data

res = []
lab = []
for i in range(25):
    print('fold ' + str(i+1))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.001, patience=1, min_lr=0.0001)
    model = make_model()
    model.fit(train_arr, tumor_fraction.loc[lucap_df.index], epochs = 5, batch_size = 20, \
              validation_data = (valid_arr, tumor_fraction.loc[crpc_df.index]))
    res.append(model.predict(valid_arr))
    lab.append(tumor_fraction.loc[crpc_df.index])
    #print(roc_plot(model.predict(val_arr), val_cancer))
    

total_res = []
for i in res:
    for j in i:
        total_res.append(j[0])
    
total_lab = []
for i in lab:
    #print(np.array(i))
    for j in np.array(i):
        total_lab.append(j[0])


res1 = []
for i in range(len(total_res)):
    res1.append([total_res[i]*100, total_lab[i]*100])
    
# Partition predictions into ground truth tumor fraction groups    
group_0 = []
group_1 = []
group_2 = []
group_3 = []
group_4 = []
group_5 = []
group_6 = []
group_7 = []
group_8 = []
group_9 = []
group_10 = []

# Append all predictions into their respective groups
for i in res1:
    if i[1] == 0:
        group_0.append(i[0])
    if i[1] == 1:
        group_1.append(i[0])
    if i[1] == 2:
        group_2.append(i[0])
    if i[1] == 3:
        group_3.append(i[0])
    if i[1] == 4:
        group_4.append(i[0])
    if i[1] == 5:
        group_5.append(i[0])
    if i[1] == 6:
        group_6.append(i[0])
    if int(i[1]) == 7:
        group_7.append(i[0])
    if i[1] == 8:
        group_8.append(i[0])
    if i[1] == 9:
        group_9.append(i[0])
    if i[1] == 10:
        group_10.append(i[0])
        
        
columns = [group_0, group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8, group_9, group_10]

fig, ax = plt.subplots()
ax.boxplot(columns)
plt.xticks(np.arange(1, 12, 1), np.arange(0, 11, 1))
plt.yticks(np.arange(0, 20, 1))
plt.xlabel('Actual Tumor Fraction (In percent out of 100)')
plt.ylabel('Cross-Validated TF Preds (In percent out of 100)')
plt.savefig('boxplot.jpg')


for i in range(len(columns)):
    print('Group ' + str(i) + ' Mean: ' + str(np.mean(columns[i])))

    
    
for i in range(len(columns)):
    print('Group ' + str(i) + ' StDev: ' + str(np.std(columns[i])))
    
    
# Create ROC Curves for ground truth tumor fraction by group
truth1 = len(group_0)* [0] + len(group_1) * [1]
roc1 = group_0 + group_1
roc_plot(roc1, truth1, 'Tf0.01_ROC_Curve.jpg')

truth2 = len(group_0)* [0] + len(group_2) * [1]
roc2 = group_0 + group_2
roc_plot(roc2, truth2, 'Tf0.02_ROC_Curve.jpg')

truth3 = len(group_0)* [0] + len(group_3) * [1]
roc3 = group_0 + group_3
roc_plot(roc3, truth3, 'Tf0.03_ROC_Curve.jpg')

truth4 = len(group_0)* [0] + len(group_4) * [1]
roc4 = group_0 + group_4
roc_plot(roc4, truth4, 'Tf0.04_ROC_Curve.jpg')

truth5 = len(group_0)* [0] + len(group_5) * [1]
roc5 = group_0 + group_5
roc_plot(roc5, truth5, 'Tf0.05_ROC_Curve.jpg')

truth6 = len(group_0)* [0] + len(group_6+group_7+group_8+group_9+group_10) * [1]
roc6 = group_0 + group_6+group_7+group_8+group_9+group_10
roc_plot(roc6, truth6,'Tf>=0.06_ROC_Curve.jpg')
