CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/


Command:
python cifar10_train.py
tensorboard --logdir=/tmp/cifar10_train
python cifar10_eval.py
python cifar10_multi_gpu_train.py --num_gpus=2

