# CNN-Image-Classifier

### 1.Prerequisite

Clone repo

```
git clone https://github.com/RagingPsyduck/CNN-Image-Classifier.git
```

Download files

1. Image set : [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. The weights for AlexNet : [bvlc_alexnet.npy](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)

Put these twos file into the work directory

Create Directory

```
mkdir cifarOuput
cd cifarOuput
mkdir checkpoints
mkdir tensorboard
```

### 2.Training

Training basic cnn uses CIFAR10.py

Training AlexNet uses CIFAR10AlexNet.py