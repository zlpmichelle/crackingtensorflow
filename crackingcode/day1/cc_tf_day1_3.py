
# coding: utf-8

# In[1]:


import tensorflow as tf
a = tf.truncated_normal([16, 128, 128, 3])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.shape(a))


# In[4]:


b = tf.reshape(a, [16, 49152])
sess.run(tf.shape(b))


# In[ ]:





# In[ ]:




