import tensorflow as tf
import numpy as np

def AlexNetCIFAR10(X,preTrainedData):
    # Layer 01: Convolutional.
    # Set Layer Parameters & Network
    W1 = tf.Variable(preTrainedData["conv1"][0])
    B1 = tf.Variable(preTrainedData["conv1"][1])
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
    W2 = tf.Variable(preTrainedData["conv2"][0])
    B2 = tf.Variable(preTrainedData["conv2"][1])
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
    W3 = tf.Variable(preTrainedData["conv3"][0])
    B3 = tf.Variable(preTrainedData["conv3"][1])
    Conv3 = tf.nn.conv2d(Maxpool2, W3, [1, 1, 1, 1], padding='SAME')
    Conv3 = tf.reshape(tf.nn.bias_add(Conv3, B3), [-1] + Conv3.get_shape().as_list()[1:])

    # Set Activation
    Conv3 = tf.nn.relu(Conv3)

    # Layer 04: Convolutional.
    # Set Layer Parameters & Network
    W4 = tf.Variable(preTrainedData["conv4"][0])
    B4 = tf.Variable(preTrainedData["conv4"][1])
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
    W5 = tf.Variable(preTrainedData["conv5"][0])
    B5 = tf.Variable(preTrainedData["conv5"][1])
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
    W6 = tf.Variable(preTrainedData["fc6"][0])
    B6 = tf.Variable(preTrainedData["fc6"][1])
    Flat = tf.reshape(Maxpool5, [-1, int(np.prod(Maxpool5.get_shape()[1:]))])
    N6 = tf.nn.relu(tf.matmul(Flat, W6) + B6)

    # Layer 07: Fully Connected.
    W7 = tf.Variable(preTrainedData["fc7"][0])
    B7 = tf.Variable(preTrainedData["fc7"][1])
    N7 = tf.nn.relu(tf.matmul(N6, W7) + B7)

    # Return Last Layer
    return N7

