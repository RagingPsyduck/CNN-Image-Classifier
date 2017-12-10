import pickle
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import time
import AlexNet as AlexNet


LabelNames=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
CIFARPATH= 'cifar-10-batches-py'

# Specify Number of Batches
numOfBatches=5
# Initiate Training Data
XTrain,YTrain=[],[]

# Load Batch Data in Loops
for BatchID in range(1, numOfBatches+1):
    with open(CIFARPATH+ '/data_batch_'+str(BatchID), mode='rb') as File:
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


with open(CIFARPATH+ '/test_batch', mode='rb') as File:
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

Epochs=20
BatchSize=128


preTrainedData=np.load('bvlc_alexnet.npy', encoding='bytes').item()

# DEFINE ARCHITECTURE
# Set a Placeholder
Features=tf.placeholder(tf.float32,(None,32,32,3))
Labels=tf.placeholder(tf.int64,None)
Resized=tf.image.resize_images(Features,(227,227))



N7=AlexNet.AlexNetCIFAR10(Resized,preTrainedData)
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