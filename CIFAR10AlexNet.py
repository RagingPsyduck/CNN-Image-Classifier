import pickle
import os
import pickle
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks_cwt
import random
from sklearn.utils import shuffle
import csv
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import glob

LabelNames=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
# Load Folder Path
DatasetFolderPath='Cifar-10-Batches-Py'

# Specify Number of Batches
NumBatches=5

# Initiate Training Data
XTrain,YTrain=[],[]

# Load Batch Data in Loops
for BatchID in range(1,NumBatches+1):
    with open(DatasetFolderPath+'/data_batch_'+str(BatchID),mode='rb') as File:
        Batch=pickle.load(File,encoding='latin1')
    if (BatchID==1):
        XTrain=Batch['data'].reshape((len(Batch['data']),3,32,32)).transpose(0, 2, 3, 1)
        YTrain=Batch['labels']
    else:
        XTrainTemp,YTrainTemp=[],[]
        XTrainTemp=Batch['data'].reshape((len(Batch['data']),3,32,32)).transpose(0, 2, 3, 1)
        YTrainTemp=Batch['labels']
        XTrain=np.concatenate((XTrain,XTrainTemp),axis=0)
        YTrain=np.concatenate((YTrain,YTrainTemp),axis=0)

# Assert to Ensure Equal Size of Input & Output Data
assert(len(XTrain)==len(YTrain))

# Number of Unique Classes and Labels
NumClass=len(set(YTrain))

# Print Data Characteristics
print("Training Set:{} Samples".format(len(XTrain)))
print("Image Shape:{}".format(XTrain[0].shape))
print('Number of Classes: {}'.format(dict(zip(*np.unique(YTrain,return_counts=True)))))
print('First 20 Labels: {}'.format(YTrain[:20]))


with open(DatasetFolderPath+'/test_batch',mode='rb') as File:
    Batch=pickle.load(File, encoding='latin1')
# load the training data
XTest=Batch['data'].reshape((len(Batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
YTest=Batch['labels']




XTrain,YTrain=shuffle(XTrain,YTrain)
XTest,YTest=shuffle(XTest,YTest)
print('Training and Testing Data Shuffled')



import tensorflow as tf
from tensorflow.contrib.layers import flatten
import time

# DEFINE ARCHITECTURE
# Set Epochs and Batch Size
Epochs=20
BatchSize=128

# DEFINE ARCHITECTURE
# Load Pre-Trained Network
NetData=np.load("bvlc-alexnet.npy",encoding='bytes').item()
print('Pre-trained Network Loaded!')


# DEFINE ARCHITECTURE
# Set a Placeholder
Features=tf.placeholder(tf.float32,(None,32,32,3))
Labels=tf.placeholder(tf.int64,None)
Resized=tf.image.resize_images(Features,(227,227))
print('Set Placeholder!')