'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random

np.set_printoptions(threshold=np.inf)

# ====================
#  EXAMPLE TOY DATA GENERATOR
# ====================
'''
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000, fileData=None, fileLabel=None):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
 #           fsee.write(str(len) + "\n")
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])

            for item in self.data:
                fileData.write("%s\n" % item)

            for item in self.labels:
                fileLabel.write("%s\n" % item)
            
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
'''
#=============================
#READS DATA
class ToySequenceData(object):
   
    def __init__(self, fileName, seqMaxLen):
        self.data = []
        self.labels = []
        self.seqlen = []
        feature = open(fileName + "_features.csv", "r")
        label = open(fileName + "_labels.csv", "r")
        with feature as feature_file:
            with label as label_file:
                for line in feature:
                    fea = line
                    lab = label.readline()
                    length = (len(lab))/4
                    if(length%1!=0):
                        print("ERROR: LINE FORMAT")
                        sys.exit(1)
                    length = int(length)
                    self.seqlen.append(length)
                    fea = fea.split(',')
                    lab = lab.split(',')
                    feat = []
                    labl = []
                    for i in range(length - 1):
                        feat.append([float(fea[i])])
                        labl.append(float(lab[i]))
                    #print("FIND ODD? ", labl, length)
                    #print(lab)
                    feat.append([float(fea[length-1][:-1])])
                    labl.append(float(lab[length-1][:-1]))

                    for i in range(seqMaxLen - int(length)):
                        feat.append([-1.0])
                        labl.append(0.0)

                    self.data.append(feat)
                    self.labels.append(labl)
                    
        self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
#'''

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 1000000 # IS VERY SLOW.
batch_size = 128
display_step = 10

# Network Parameters
seq_max_len = 10 # Sequence max length
n_hidden = 64 # hidden layer num of features
#n_classes = 2 # linear sequence or not - Use for example
n_classes = seq_max_len # Use for data

#These are for input files
trainset = ToySequenceData("0622_train", seq_max_len)
testset = ToySequenceData("0622_test", seq_max_len)

'''
#Writes randomly generated sample data to files
trainData = open("0train.txt", "w")
trainLabel = open("0trainLabel.txt", "w")
testData = open("0test.txt", "w")
testLabel = open("0testLabel.txt", "w")

#These run example data
trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len, fileData=trainData, fileLabel=trainLabel)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len, fileData=testData, fileLabel=testLabel)
'''

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        #print("TESTING: ", batch_seqlen)
        #print("TESTING: ", batch_x)
        #print("TESTING: ", batch_y)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))

    f1 = open("0dnn_test_predictions_raw.txt", 'w')
#    prediction = tf.argmax(y,1)
#    print(prediction.eval(feed_dict={x: test_data, seqlen: test_seqlen}))
    f1.write(np.array2string(sess.run(tf.argmax(pred,1),feed_dict={x: test_data, seqlen: test_seqlen})))
