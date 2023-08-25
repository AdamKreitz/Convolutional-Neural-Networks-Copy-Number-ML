# Convolutional Neural Networks for Early Cancer Detection
### Repo containing scripts for implementing Convolutional Neural Networks based on Copy Number information from Cell-Free DNA

Contains Deep Learning approaches for analyzing copy number information from a range of datasets to create both a classifier for instances of cancer as well as a regressor for tumor fraction

#### Methodology
  
       
<img width="901" alt="Screen Shot 2023-02-02 at 11 23 46 AM" src="https://user-images.githubusercontent.com/53357910/216429718-01620360-2e43-4f32-ac14-3ea0d97ec7c4.png">


###

Imports:

Several libraries are imported, which are essential for data processing, visualization, and machine learning:
numpy and pandas: Used for numerical and data processing respectively.
matplotlib.pyplot: Used for plotting graphs.
os: Provides functions for interacting with the operating system.
seaborn: Used for statistical data visualization.
sklearn: Machine learning library. Specifically, functions for splitting datasets and metrics are imported.
tensorflow and keras: Libraries for designing and training neural networks.
Function Definitions:

roc_plot(): This function plots the Receiver Operating Characteristic (ROC) curve, which is used to assess the performance of a binary classification model.
save_model_activity_delfi() & save_model_activity_lucas(): Both functions seem to define a Convolutional Neural Network (CNN) model, train it on respective datasets (either Delfi or Lucas), and return the ROC plot and some results.
make_model(): Defines a basic CNN model.
Loading Dataframes:

Data from various sources (e.g., Delfi Cohort, Lucas, Validation Cohorts) are loaded into dataframes using pandas.
Reshaping Input Matrices:

The data matrices for Delfi and Lucas are reshaped to be fit into the CNNs.
Validation:

The model seems to be validated on a dataset named 'validation'. The CNN model is trained 50 times on the 'lucas' dataset and then validated on the 'validation' dataset. The predictions from each training are stored and later used to plot a ROC curve for the validation set.
