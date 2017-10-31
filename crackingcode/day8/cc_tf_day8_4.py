
# coding: utf-8

# In[ ]:


# Frequently Asked Questions

# Using `Session.run()`.
sess = tf.Session()
c = tf.constant(5.0)
print(sess.run(c))

# Using `Tensor.eval()`.
c = tf.constant(5.0)
with tf.Session():
  print(c.eval())


