
# coding: utf-8

# In[3]:


import numpy as np

num_points = 1000
vectors_set = []
for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])
    
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


# In[5]:


import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()


# In[7]:


import tensorflow as tf
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b


# In[8]:


loss = tf.reduce_mean(tf.square(y - y_data))


# In[9]:


optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# In[11]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[20]:


for step in xrange(8):
    sess.run(train)
print step, sess.run(W), sess.run(b)


# In[15]:


plt.plot(x_data, y_data, 'ro', label='New data')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()


# In[31]:


for step in xrange(8):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))


# In[32]:


for step in xrange(8):
    sess.run(train)
    print(step, sess.run(loss))


# In[34]:


for step in xrange(8):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    print(step, sess.run(loss))
    
    # graphic display
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('y')
    plt.legend()
    plt.show()


# In[51]:


row = 1000
col = 2
vectors_sets = []
for i in xrange(row):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_sets.append([x1, y1])
vectors = tf.constant(vectors_sets)
extended_vectors = tf.expand_dims(vectors, 0)
print extended_vectors.get_shape()


# In[ ]:




