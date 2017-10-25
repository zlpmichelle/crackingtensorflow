
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[5]:


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())


# In[7]:


y = tf.matmul(x, W) + b


# In[8]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))


# In[9]:


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[10]:


for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})


# In[11]:


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))


# In[12]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[13]:


print(accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))


# In[14]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


# In[15]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# In[16]:


W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])


# In[17]:


x_image = tf.reshape(x, [-1, 28,28,1])


# In[18]:


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))
h_pool1 = max_pool_2x2(h_conv1)


# In[19]:


W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) 
h_pool2 = max_pool_2x2(h_conv2)


# In[20]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


# In[22]:


h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[ ]:




