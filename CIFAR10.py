import tensorflow as tf
import CIFARHelper as CIFARHelper

LEARNING_RATE = 0.001 # 0.001
STEP = 5000 # 5000

CIFARPATH = 'cifar-10-batches-py/'
FILEWRITER_PATH = "./cifarOutput/tensorboard"
CHECKPOINT_PATH = "./cifarOutput/checkpoints"
dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
inputs = [0, 1, 2, 3, 4, 5, 6]

for i, dir in zip(inputs, dirs):
    inputs[i] = CIFARHelper.unpickle(CIFARPATH + dir)

batchMeta = inputs[0]
batch1 = inputs[1]
batch2 = inputs[2]
batch3 = inputs[3]
batch4 = inputs[4]
batch5 = inputs[5]
testBatch = inputs[6]

ch = CIFARHelper.CifarHelper(batch1=batch1,batch2=batch2,batch3=batch3,batch4=batch4,batch5=batch5,testBatch=testBatch)
ch.set_up_images()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
# Dropout
hold_prob = tf.placeholder(tf.float32)

conv1 = CIFARHelper.convLayer(x, shape=[4, 4, 3, 32])
conv1Pooling = CIFARHelper.pool2by2(conv1)
conv2 = CIFARHelper.convLayer(conv1Pooling, shape=[4, 4, 32, 64])
conv2Pooling = CIFARHelper.pool2by2(conv2)
conv2Flat = tf.reshape(conv2Pooling, [-1, 8 * 8 * 64])
fullConnectedLayer = tf.nn.relu(CIFARHelper.normalFullLayer(conv2Flat, 1024))
full_one_dropout = tf.nn.dropout(fullConnectedLayer, keep_prob=hold_prob)
y_pred = CIFARHelper.normalFullLayer(full_one_dropout, 10)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()

writer = tf.summary.FileWriter(FILEWRITER_PATH)

with tf.name_scope('Accuracy'):
    correctPrediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar('cross_entropy', cross_entropy)
mergedSummary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    for i in range(STEP):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
        if i % 5 == 0:
            summary, acc = sess.run([mergedSummary, accuracy], feed_dict={x: ch.test_images, y_true: ch.test_labels, hold_prob: 1.0})
            writer.add_summary(summary, i)
            print('Step {} , Accuracy is:{:.4f}'.format(i,sess.run(accuracy, feed_dict={x: ch.test_images, y_true: ch.test_labels, hold_prob: 1.0})))
