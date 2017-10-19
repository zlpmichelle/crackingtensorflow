
# coding: utf-8

# In[2]:


import tensorflow as tf
graph = tf.get_default_graph()
graph.get_operations()
for op in graph.get_operations():
    print(op.name)


# In[4]:


sess = tf.Session()
sess.close()


# In[6]:


with tf.Session() as sess:
    sess.run(f)


# In[8]:


a = tf.constant(1.0)
a
print(a)


# In[10]:


with tf.Session() as sess:
    print(sess.run(a))


# In[11]:


b = tf.Variable(2.0, name = "test_var")
b


# In[12]:


init_op = tf.global_variables_initializer()


# In[13]:


with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(b))    


# In[14]:


graph = tf.get_default_graph()
for op in graph.get_operations():
    print(op.name)


# In[15]:


a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.multiply(a, b)
feed_dict = {a:2, b:3}
with tf.Session() as sess:
    print(sess.run(y, feed_dict))


# In[16]:


w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))


# In[22]:


b = tf.Variable([10,20,30,40,50,60], name='t')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.reduce_mean(b)))


# In[26]:


a =[[0.1, 0.2, 0.3],
[20, 2, 3]]
b = tf.Variable(a, name='b')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.argmax(b, 1)))


# In[ ]:




