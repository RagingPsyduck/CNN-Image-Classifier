import pickle
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import AlexNet as AlexNet
from sklearn.model_selection import train_test_split

LabelNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
CIFARPATH = 'cifar-10-batches-py'
trainInput, trainLabel = [], []
EPOCH = 20
BATCH_SIZE = 128


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


assert (len(trainInput) == len(trainLabel))
classCount = len(set(trainLabel))


with open(CIFARPATH + '/test_batch', mode='rb') as File:
    Batch = pickle.load(File, encoding='latin1')


testInput = Batch['data'].reshape((len(Batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
testLabel = Batch['labels']

trainInput, trainLabel = shuffle(trainInput, trainLabel)
testInput, testLabel = shuffle(testInput, testLabel)

trainInput, XVal, trainLabel, YVal = train_test_split(trainInput, trainLabel, test_size=0.10, random_state=0)


def evaluate(X, Y, Sess):
    totalAcc = 0
    totalLoss = 0
    for Offset in range(0, X.shape[0], BATCH_SIZE):
        End = Offset + BATCH_SIZE
        XBatch = X[Offset:End]
        YBatch = Y[Offset:End]
        Loss, Acc = Sess.run([loss, accuracy], feed_dict={features: XBatch, labels: YBatch})
        totalLoss += (Loss * XBatch.shape[0])
        totalAcc += (Acc * XBatch.shape[0])
    return totalLoss / X.shape[0], totalAcc / X.shape[0]


initWeight = np.load('bvlc_alexnet.npy',encoding='bytes').item()

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resize = tf.image.resize_images(features, (227, 227))

x = AlexNet.train(resize, initWeight)
x = tf.stop_gradient(x)
shape = (x.get_shape().as_list()[-1], classCount)
weight = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
bias = tf.Variable(tf.zeros(classCount))
y = tf.matmul(x, weight) + bias


crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labels)
loss = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss, var_list=[weight, bias])
predict = tf.arg_max(y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, labels), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(EPOCH):
        trainInput, trainLabel = shuffle(trainInput, trainLabel)
        for offset in range(0, trainInput.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            sess.run(train, feed_dict={features: trainInput[offset:end], labels: trainLabel[offset:end]})

        _, acc = evaluate(XVal, YVal, sess)
        print("Epoch {}, Accuracy {}".format(step + 1, acc))
