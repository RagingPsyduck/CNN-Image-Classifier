import tensorflow as tf
from alexnet import AlexNet
import glob

class_name = ['cat', 'dog']



def getCorretNumberOfImages(path, type):
    count = 0
    files = glob.glob(path)
    print(files)
    totalCount = len(files)
    for file in files:
        print(file)
        pro = getTypebyImagePath(file)
        print(pro)

    return count


def getTypebyImagePath(path_image):
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[1, 227, 227, 3])
    model = AlexNet(img_resized, 0.5, 2, skip_layer='', weights_path='Default')
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, tf.train.latest_checkpoint('.'))
        saver.restore(sess, "./output/checkpoints/model_epoch5.ckpt")
        # score = model.fc8
        print(sess.run(model.fc8))
        res = sess.run(max)[0]
    return res



catCount = getCorretNumberOfImages('test/cat/*.jpg', 0)
