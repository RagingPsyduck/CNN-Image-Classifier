import pickle
import numpy as np
import tensorflow as tf



class CifarHelper():
    def __init__(self,batch1,batch2,batch3,batch4,batch5,testBatch):
        self.i = 0
        self.AllTrainBatches = [batch1, batch2, batch3, batch4, batch5]
        self.TestBatch = [testBatch]
        self.TrainingImages = None
        self.TrainingLabels = None
        self.TestImages = None
        self.TestLabels = None

    def setUpImages(self):
        self.TrainingImages = np.vstack([d[b"data"] for d in self.AllTrainBatches])
        train_len = len(self.TrainingImages)

        self.TrainingImages = self.TrainingImages.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.TrainingLabels = encode(np.hstack([d[b"labels"] for d in self.AllTrainBatches]), 10)

        self.TestImages = np.vstack([d[b"data"] for d in self.TestBatch])
        testLen = len(self.TestImages)

        self.TestImages = self.TestImages.reshape(testLen, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.TestLabels = encode(np.hstack([d[b"labels"] for d in self.TestBatch]), 10)

    def next_batch(self, batch_size):
        x = self.TrainingImages[self.i:self.i + batch_size].reshape(batch_size, 32, 32, 3)
        y = self.TrainingLabels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.TrainingImages)
        return x, y

def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def encode(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def initWeights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def initBias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def pool2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def convLayer(input_x, shape):
    W = initWeights(shape)
    b = initBias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normalFullLayer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = initWeights([input_size, size])
    b = initBias([size])
    return tf.matmul(input_layer, W) + b