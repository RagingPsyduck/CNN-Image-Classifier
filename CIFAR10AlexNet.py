
import pickle
import numpy as np
from sklearn.utils import shuffle


LabelNames=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
# Load Folder Path
DatasetFolderPath='cifar-10-batches-pgity'

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

from sklearn.cross_validation import train_test_split
XTrain,XVal,YTrain,YVal=train_test_split(XTrain,YTrain,test_size=0.10,random_state=0)
print('Training Data Randomized and Split for Validation')
print('Training Data Size:'+str(XTrain.shape))
print('Validation Data Size:'+str(XVal.shape))

import tensorflow as tf
#from tensorflow.contrib.layers import flatten
import time

# DEFINE ARCHITECTURE
# Set Epochs and Batch Size
Epochs=20
BatchSize=128

# DEFINE ARCHITECTURE
# Load Pre-Trained Network
NetData=np.load('bvlc_alexnet.npy',encoding='bytes').item()
print('Pre-trained Network Loaded!')


# DEFINE ARCHITECTURE
# Set a Placeholder
Features=tf.placeholder(tf.float32,(None,32,32,3))
Labels=tf.placeholder(tf.int64,None)
Resized=tf.image.resize_images(Features,(227,227))
print('Set Placeholder!')


# DEFINE ALEXNET ARCHITECTURE
def AlexNetCIFAR10(X):
    # Layer 01: Convolutional.
    # Set Layer Parameters & Network
    W1 = tf.Variable(NetData["conv1"][0])
    B1 = tf.Variable(NetData["conv1"][1])
    CO = 96
    CI = X.get_shape()[-1]
    assert CI % 1 == 0
    assert CO % 1 == 0
    Conv1Init = tf.nn.conv2d(X, W1, [1, 4, 4, 1], padding='SAME')
    Conv1 = tf.reshape(tf.nn.bias_add(Conv1Init, B1), [-1] + Conv1Init.get_shape().as_list()[1:])

    # Set Activation
    Conv1 = tf.nn.relu(Conv1)

    # Do Normalization
    Lrn1 = tf.nn.local_response_normalization(Conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

    # Do Pooling
    Maxpool1 = tf.nn.max_pool(Lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 02: Convolutional.
    # Set Layer Parameters & Network
    W2 = tf.Variable(NetData["conv2"][0])
    B2 = tf.Variable(NetData["conv2"][1])
    InputGroups = tf.split(Maxpool1, 2, 3)
    KernelGroups = tf.split(W2, 2, 3)
    Convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    OutputGroups = [Convolve(i, k) for i, k in zip(InputGroups, KernelGroups)]
    Conv2 = tf.concat(OutputGroups, 3)
    Conv2 = tf.reshape(tf.nn.bias_add(Conv2, B2), [-1] + Conv2.get_shape().as_list()[1:])

    # Set Activation
    Conv2 = tf.nn.relu(Conv2)

    # Do Normalization
    Lrn2 = tf.nn.local_response_normalization(Conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1)

    # Do Pooling
    Maxpool2 = tf.nn.max_pool(Lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 03: Convolutional.
    # Set Layer Parameters & Network
    W3 = tf.Variable(NetData["conv3"][0])
    B3 = tf.Variable(NetData["conv3"][1])
    Conv3 = tf.nn.conv2d(Maxpool2, W3, [1, 1, 1, 1], padding='SAME')
    Conv3 = tf.reshape(tf.nn.bias_add(Conv3, B3), [-1] + Conv3.get_shape().as_list()[1:])

    # Set Activation
    Conv3 = tf.nn.relu(Conv3)

    # Layer 04: Convolutional.
    # Set Layer Parameters & Network
    W4 = tf.Variable(NetData["conv4"][0])
    B4 = tf.Variable(NetData["conv4"][1])
    InputGroups = tf.split(Conv3, 2, 3)
    KernelGroups = tf.split(W4, 2, 3)
    Convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    OutputGroups = [Convolve(i, k) for i, k in zip(InputGroups, KernelGroups)]
    Conv4 = tf.concat(OutputGroups, 3)
    Conv4 = tf.reshape(tf.nn.bias_add(Conv4, B4), [-1] + Conv4.get_shape().as_list()[1:])

    # Set Activation
    Conv4 = tf.nn.relu(Conv4)

    # Layer 05: Convolutional.
    # Set Layer Parameters & Network
    W5 = tf.Variable(NetData["conv5"][0])
    B5 = tf.Variable(NetData["conv5"][1])
    InputGroups = tf.split(Conv4, 2, 3)
    KernelGroups = tf.split(W5, 2, 3)
    Convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    OutputGroups = [Convolve(i, k) for i, k in zip(InputGroups, KernelGroups)]
    Conv5 = tf.concat(OutputGroups, 3)
    Conv5 = tf.reshape(tf.nn.bias_add(Conv5, B5), [-1] + Conv5.get_shape().as_list()[1:])

    # Set Activation
    Conv5 = tf.nn.relu(Conv5)

    # Do Pooling
    Maxpool5 = tf.nn.max_pool(Conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 06: Fully Connected.
    W6 = tf.Variable(NetData["fc6"][0])
    B6 = tf.Variable(NetData["fc6"][1])
    Flat = tf.reshape(Maxpool5, [-1, int(np.prod(Maxpool5.get_shape()[1:]))])
    N6 = tf.nn.relu(tf.matmul(Flat, W6) + B6)

    # Layer 07: Fully Connected.
    W7 = tf.Variable(NetData["fc7"][0])
    B7 = tf.Variable(NetData["fc7"][1])
    N7 = tf.nn.relu(tf.matmul(N6, W7) + B7)

    # Return Last Layer
    return N7

N7=AlexNetCIFAR10(Resized)
N7=tf.stop_gradient(N7)
Shape=(N7.get_shape().as_list()[-1],NumClass)
W8=tf.Variable(tf.truncated_normal(Shape,stddev=1e-2))
B8=tf.Variable(tf.zeros(NumClass))
Logits=tf.nn.xw_plus_b(N7,W8,B8)

# DEFINE ALEXNET ARCHITECTURE
# Set Training Pipeline
CrossEntropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Logits,labels=Labels)
LossOp=tf.reduce_mean(CrossEntropy)
Opt=tf.train.AdamOptimizer()
TrainOp=Opt.minimize(LossOp,var_list=[W8,B8])
InitOp=tf.global_variables_initializer()
Preds=tf.arg_max(Logits,1)
AccuracyOp=tf.reduce_mean(tf.cast(tf.equal(Preds,Labels),tf.float32))


# MODEL EVALUATION
# Initialize Evaluation
def Evaluate(X, Y, Sess):
    TotalAcc = 0
    TotalLoss = 0
    for Offset in range(0, X.shape[0], BatchSize):
        End = Offset + BatchSize
        XBatch = X[Offset:End]
        YBatch = Y[Offset:End]
        Loss, Acc = Sess.run([LossOp, AccuracyOp], feed_dict={Features: XBatch, Labels: YBatch})
        TotalLoss += (Loss * XBatch.shape[0])
        TotalAcc += (Acc * XBatch.shape[0])

    # Return Loss and Accuracy
    return TotalLoss / X.shape[0], TotalAcc / X.shape[0]

with tf.Session() as SessMajor:
    SessMajor.run(InitOp)

    for i in range(Epochs):
        XTrain,YTrain=shuffle(XTrain,YTrain)
        T0=time.time()
        for Offset in range(0,XTrain.shape[0],BatchSize):
            End=Offset+BatchSize
            SessMajor.run(TrainOp,feed_dict={Features:XTrain[Offset:End],Labels:YTrain[Offset:End]})

        # val_loss, val_acc = eval_on_data(XVal,YVal,sess)
        ValLoss,ValAcc=Evaluate(XVal,YVal,SessMajor)
        print("Epoch",i+1)
        print("Time: %.3f seconds"%(time.time()-T0))
        print("Validation Loss =",ValLoss)
        print("Validation Accuracy =",ValAcc)
        print("")