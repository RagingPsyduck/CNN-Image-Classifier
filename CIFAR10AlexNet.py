import pickle
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import AlexNet as AlexNet
from sklearn.model_selection import train_test_split
from random import randint

LabelNames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
CIFARPATH = 'cifar-10-batches-py'
FILEWRITER_PATH = "./cifarOutput/tensorboard"
CHECKPOINT_PATH = "./cifarOutput/checkpoints"

trainInput, trainLabel = [], []
LEARNING_RATE = 0.001
EPOCH = 20
BATCH_SIZE = 200
TEST_SIZE = 100
DROPOUT = 0.5 #0.5

for i in range(1, 6):
    with open(CIFARPATH + '/data_batch_' + str(i), mode='rb') as File:
        Batch = pickle.load(File, encoding='latin1')
    if i == 1:
        trainInput = Batch['data'].reshape((len(Batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        trainLabel = Batch['labels']
    else:
        trainInputTemp, trainLabelTemp = [], []
        trainInputTemp = Batch['data'].reshape((len(Batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        trainLabelTemp = Batch['labels']
        trainInput = np.concatenate((trainInput, trainInputTemp), axis=0)
        trainLabel = np.concatenate((trainLabel, trainLabelTemp), axis=0)

classCount = len(set(trainLabel))

with open(CIFARPATH + '/test_batch', mode='rb') as File:
    Batch = pickle.load(File, encoding='latin1')

testInput = Batch['data'].reshape((len(Batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
testLabel = Batch['labels']
trainInput, trainLabel = shuffle(trainInput, trainLabel)
testInput, testLabel = shuffle(testInput, testLabel)
trainInput, XVal, trainLabel, YVal = train_test_split(trainInput, trainLabel, test_size=0.10, random_state=0)


initWeight = np.load('bvlc_alexnet.npy',encoding='bytes').item()
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resize = tf.image.resize_images(features, (227, 227))
lastLayer = AlexNet.train(resize, initWeight)
lastLayer = tf.stop_gradient(lastLayer)
shape = (lastLayer.get_shape().as_list()[-1], classCount)
weight = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
bias = tf.Variable(tf.zeros(classCount))
y = tf.matmul(lastLayer, weight) + bias

crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labels)
loss = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss, var_list=[weight, bias])
predict = tf.arg_max(y, 1)


writer = tf.summary.FileWriter(FILEWRITER_PATH)
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, labels), tf.float32))

tf.summary.scalar("accuracy", accuracy)
mergedSummary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(EPOCH):
        trainInput, trainLabel = shuffle(trainInput, trainLabel)
        randStart = randint(0,4)
        randStart = randStart * 100
        summary, acc,_ = sess.run([mergedSummary, accuracy, train],feed_dict={features: testInput[randStart:randStart+TEST_SIZE], labels: testLabel[randStart:randStart+TEST_SIZE]})
        print("Epoch {}, Accuracy {}".format(step + 1, acc))
        writer.add_summary(summary, step)
        for start in range(0, trainInput.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train, feed_dict={features: trainInput[start:end], labels: trainLabel[start:end]})