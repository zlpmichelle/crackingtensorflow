
# coding: utf-8

# In[1]:


# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py
import tensorflow as tf

# download and read in the data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[2]:


import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])


# In[7]:


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[8]:


y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[30]:


y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])))
# following is much better
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# In[31]:


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[32]:


sess = tf.InteractiveSession()


# In[33]:


tf.global_variables_initializer().run()


# In[34]:


for _ in range(1000):
    # SGD:  stochastic gradient descent
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})


# In[35]:


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))


# In[36]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[37]:


print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:




