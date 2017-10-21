
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import datetime


# In[13]:


A = np.random.rand(1, 100).astype('int32')
B = np.random.rand(1, 100).astype('int32')

n = 10


# In[11]:


c1 = []
c2 = []


# In[17]:


def matpow(M, n):
    if n < 1:
        return M
    else:
        return tf.malmul(M, matpow(M, n-1))


# In[21]:


with tf.device('/gpu:0'):
    a = tf.constant(A)
    b = tf.constant(B)
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))
    
with tf.device('/cpu:0'):
    sum = tf.add_n(c1)
    t1_1 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(sum)
t2_1 = datetime.datetime.now()


# In[19]:


with tf.device('/gpu:0'):
    # compute A^n and store result in c2
    a = tf.constant(A)
    c2.append(matpow(a, n))
    
with tf.device('/gpu:1'):
    # compute B^n and store result in c2
    b = tf.constant(B)
    c2.append(matpow(b, n))
    
with tf.device('/cpu:0'):
    # Addition of all elements in c2, i.e. A^n + B^n
    sum = tf.add_n(c2)
    t1_2 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # runs the op.
    sess.run(sum)
    t2_2 = datatime.datetime.now()
    


# In[22]:


print "Single GPU computation time: " + str(t2_1 - t1_1)
print "Multi GPU computation time: " + str(t2_2 - t1_2)


# In[ ]:




