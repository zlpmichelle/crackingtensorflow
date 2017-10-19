
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

trainX = np.linspace(-1, 1, 101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33


# In[12]:


X = tf.placeholder("float")
Y = tf.placeholder("float")
# Linear regression model is y_model=w*x and we have to calculate the value of w through our model. 
# Lets' initialize w to 0 and create a model to solve this problem.
w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X, w)

cost = (tf.pow(Y - y_model, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


# In[13]:


# create init_op to initialize all variables
init = tf.global_variables_initializer()


# In[24]:


with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: x, Y: y})
            #print("round " + str(i) + ": " + str(sess.run(w)))
    print(sess.run(w))


# In[26]:


with tf.Session() as sess:
    sess.run(init)
    print(sess.run(w))


# In[ ]:




