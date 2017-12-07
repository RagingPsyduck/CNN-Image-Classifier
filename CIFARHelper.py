import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

class CifarHelper():
    def __init__(self,batch1,batch2,batch3,batch4,batch5,testBatch):
        self.i = 0
        self.all_train_batches = [batch1, batch2, batch3, batch4, batch5]
        self.test_batch = [testBatch]
        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out